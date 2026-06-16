####################################################################################################
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
####################################################################################################

import os
import time
from dataclasses import dataclass
from typing import Dict, Optional, Protocol, Tuple

import numpy as np
import onnxruntime as ort
from PIL import Image

import qaic

from .preprocess import MaskRCNNImage
from .postprocess import PostprocessConfig, PostprocessResult, postprocess_detections

@dataclass(frozen=True)
class LatencyStats:
    mean_ms: float
    p50_ms: float
    p90_ms: float
    p99_ms: float


@dataclass(frozen=True)
class PredictResult:
    post: PostprocessResult
    latencies: Dict[str, LatencyStats]


class _MaskRcnnPredictor(Protocol):
    def predict(
        self,
        image_path: str,
        *,
        score_thresh: float = 0.05,
        nms_thresh: float = 0.5,
        topk: int = 100,
        warmup: int = 1,
        runs: int = 10,
    ) -> PredictResult: ...


class MaskRCNNOnnxCPU:
    """Split-ONNX runner for Mask R-CNN.

    Expected files in model_dir:
      - model_backbone_fpn.onnx
      - model_rpn.onnx
      - model_roi_heads.onnx

    Postprocess (decode+NMS) runs in Python.
    """

    def __init__(self, model_dir: str, providers: Optional[list[str]] = None):
        self.model_dir = model_dir
        self.backbone_path = os.path.join(model_dir, "model_backbone_fpn.onnx")
        self.rpn_path = os.path.join(model_dir, "model_rpn.onnx")
        self.roi_path = os.path.join(model_dir, "model_roi_heads.onnx")

        # Validate that required ONNX files exist
        missing = [p for p in [self.backbone_path, self.rpn_path, self.roi_path] if not os.path.exists(p)]
        if missing:
            rel_missing = [os.path.relpath(p, model_dir) for p in missing]
            raise FileNotFoundError(
                "Missing ONNX model files in MaskRCNNOnnxCPU model_dir. "
                f"Missing: {', '.join(rel_missing)}. "
                "Please export models first and try again."
            )

        if providers is None:
            providers = ["CPUExecutionProvider"]

        self.sess_backbone = ort.InferenceSession(self.backbone_path, providers=providers)
        self.sess_rpn = ort.InferenceSession(self.rpn_path, providers=providers)
        self.sess_roi = ort.InferenceSession(self.roi_path, providers=providers)

    @staticmethod
    def _latency_stats(samples_ms: list[float]) -> LatencyStats:
        a = np.asarray(samples_ms, dtype=np.float64)
        return LatencyStats(
            mean_ms=float(a.mean()),
            p50_ms=float(np.percentile(a, 50)),
            p90_ms=float(np.percentile(a, 90)),
            p99_ms=float(np.percentile(a, 99)),
        )

    @staticmethod
    def _time_run(sess: ort.InferenceSession, feed: dict, warmup: int, runs: int) -> Tuple[list, LatencyStats]:
        for _ in range(max(0, warmup)):
            _ = sess.run(None, feed)

        samples = []
        out = None
        for _ in range(max(1, runs)):
            t0 = time.perf_counter()
            out = sess.run(None, feed)
            t1 = time.perf_counter()
            samples.append((t1 - t0) * 1000.0)

        return out, MaskRCNNOnnxCPU._latency_stats(samples)

    def predict(
        self,
        image_path: str,
        *,
        score_thresh: float = 0.05,
        nms_thresh: float = 0.5,
        topk: int = 100,
        warmup: int = 1,
        runs: int = 10,
    ) -> PredictResult:
        """Run the ONNX pipeline.

        Args:
            image_path: path to an RGB image (will be resized to 800x1216 for this export)
        """
        lat: Dict[str, LatencyStats] = {}

        image_nchw_fp32 = MaskRCNNImage.load(image_path, batch=1)  # [1, 3,800,1216]

        # backbone
        (p2, p3, p4, p5, p6), lat_backbone = self._time_run(
            self.sess_backbone,
            {"image": image_nchw_fp32},
            warmup=warmup,
            runs=runs,
        )
        lat["backbone_fpn"] = lat_backbone

        # rpn
        rpn_inputs = {i.name: t for i, t in zip(self.sess_rpn.get_inputs(), [p2, p3, p4, p5, p6])}
        (proposals,), lat_rpn = self._time_run(self.sess_rpn, rpn_inputs, warmup=warmup, runs=runs)
        lat["rpn"] = lat_rpn

        # roi
        roi_inputs = {i.name: t for i, t in zip(self.sess_roi.get_inputs(), [p2, p3, p4, p5, proposals])}
        (cls_logits, bbox_deltas, mask_logits), lat_roi = self._time_run(
            self.sess_roi, roi_inputs, warmup=warmup, runs=runs
        )
        lat["roi_heads"] = lat_roi

        # python postprocess
        pp_cfg = PostprocessConfig(
            num_classes=80,
            score_thresh=score_thresh,
            nms_thresh=nms_thresh,
            topk=topk,
        )

        for _ in range(max(0, warmup)):
            _ = postprocess_detections(
                proposals_xyxy=proposals,
                cls_logits=cls_logits,
                bbox_deltas=bbox_deltas,
                mask_logits=mask_logits,
                image_hw=(800, 1216),
                cfg=pp_cfg,
            )

        pp_samples = []
        post = None
        for _ in range(max(1, runs)):
            t0 = time.perf_counter()
            post = postprocess_detections(
                proposals_xyxy=proposals,
                cls_logits=cls_logits,
                bbox_deltas=bbox_deltas,
                mask_logits=mask_logits,
                image_hw=(800, 1216),
                cfg=pp_cfg,
            )
            t1 = time.perf_counter()
            pp_samples.append((t1 - t0) * 1000.0)

        lat["python_postprocess"] = self._latency_stats(pp_samples)

        assert post is not None
        return PredictResult(post=post, latencies=lat)


class MaskRCNNQaic:
    """QAIC-based implementation.

    Backbone+FPN and ROI heads run on the accelerator via qaic.Session.
    RPN and postprocess run on CPU using ONNXRuntime and Python, respectively.

    Expected directory layout (example):
      model_dir/
        aic_model_backbone_fpn/programqpc.bin
        aic_model_roi_heads/programqpc.bin
        model_rpn.onnx
    """

    def __init__(self, model_dir: str, providers: Optional[list[str]] = None, device_id: Optional[int] = 1):
        if qaic is None:
            raise ImportError("qaic module not available; install the QAIC SDK to use MaskRCNNQaic.")

        self.model_dir = model_dir
        self.providers = providers or ["QAicExecutionProvider"]

        # QAIC binaries
        self.backbone_prog = os.path.join(model_dir, "aic_model_backbone_fpn", "programqpc.bin")
        self.roi_prog = os.path.join(model_dir, "aic_model_roi_heads", "programqpc.bin")

        # RPN ONNX (CPU)
        self.rpn_onnx = os.path.join(model_dir, "model_rpn.onnx")

        # Validate that required QAIC and ONNX files exist
        required_paths = [self.backbone_prog, self.roi_prog, self.rpn_onnx]
        missing = [p for p in required_paths if not os.path.exists(p)]
        if missing:
            rel_missing = [os.path.relpath(p, model_dir) for p in missing]
            raise FileNotFoundError(
                "Missing QPC model binary or ONNX model in MaskRCNNQaic model_dir. "
                f"Missing: {', '.join(rel_missing)}. "
                "Please export models first and try again."
            )

        # QAIC sessions: adapt these calls to your SDK if needed
        self.q_sess_backbone = qaic.Session(self.backbone_prog,
                                      num_activations=1,
                                      set_size=1,
                                      dev_id=device_id,
                                      oversubscription_name='group1')
        self.q_sess_roi = qaic.Session(self.roi_prog,
                                      num_activations=1,
                                      set_size=1,
                                      dev_id=device_id,
                                      oversubscription_name='group1')

        # RPN session via ORT CPU
        self.sess_rpn = ort.InferenceSession(self.rpn_onnx, providers=["CPUExecutionProvider"])

    @staticmethod
    def _latency_stats(samples_ms: list[float]) -> LatencyStats:
        a = np.asarray(samples_ms, dtype=np.float64)
        return LatencyStats(
            mean_ms=float(a.mean()),
            p50_ms=float(np.percentile(a, 50)),
            p90_ms=float(np.percentile(a, 90)),
            p99_ms=float(np.percentile(a, 99)),
        )

    def _time_qp(self, sess, feed, runs: int) -> Tuple[list, LatencyStats]:
        """Time a QAIC session.

        `feed` must be adapted to match the qaic.Session.run() API.
        Here we assume sess.run(feed) returns a list of numpy arrays.
        """
        samples = []
        out = None
        for _ in range(max(1, runs)):
            t0 = time.perf_counter()
            out = sess.run(feed)
            t1 = time.perf_counter()
            samples.append((t1 - t0) * 1000.0)
        return out, self._latency_stats(samples)

    def predict(
        self,
        image_path: str,
        *,
        score_thresh: float = 0.05,
        nms_thresh: float = 0.5,
        topk: int = 100,
        warmup: int = 1,
        runs: int = 10,
    ) -> PredictResult:
        lat: Dict[str, LatencyStats] = {}

        # preprocess image
        _, input_type = self.q_sess_backbone.model_input_shape_dict['image']
        image_chw_fp32 = MaskRCNNImage.load(image_path)  # [3,800,1216]
        image_nchw = image_chw_fp32[np.newaxis, ...].astype(input_type)  # [1,3,800,1216]

        # --- backbone_fpn on QAIC ---
        backbone_feed = {"image": image_nchw}
        for _ in range(max(0, warmup)):
            _ = self.q_sess_backbone.run(backbone_feed)

        backbone_out, lat_backbone = self._time_qp(self.q_sess_backbone, backbone_feed, runs=runs)
        p2, p3, p4, p5, p6 = [backbone_out[k] for k in ("p2", "p3", "p4", "p5", "p6")]
        lat["backbone_fpn"] = lat_backbone

        # --- RPN on CPU (ONNXRuntime) ---
        rpn_inputs = {i.name: t for i, t in zip(self.sess_rpn.get_inputs(), [p2, p3, p4, p5, p6])}
        for _ in range(max(0, warmup)):
            _ = self.sess_rpn.run(None, rpn_inputs)

        rpn_samples = []
        proposals = None
        for _ in range(max(1, runs)):
            t0 = time.perf_counter()
            (proposals,) = self.sess_rpn.run(None, rpn_inputs)
            t1 = time.perf_counter()
            rpn_samples.append((t1 - t0) * 1000.0)
        lat["rpn"] = self._latency_stats(rpn_samples)
        assert proposals is not None

        # --- ROI heads on QAIC ---
        roi_feed = {"p2": p2, "p3": p3, "p4": p4, "p5": p5, "proposals": proposals}
        for _ in range(max(0, warmup)):
            _ = self.q_sess_roi.run(roi_feed)

        roi_out, lat_roi = self._time_qp(self.q_sess_roi, roi_feed, runs=runs)
        cls_logits, bbox_deltas, mask_logits = [roi_out[k] for k in ("cls_logits", "bbox_deltas", "mask_logits")]

        lat["roi_heads"] = lat_roi

        # --- python postprocess ---
        pp_cfg = PostprocessConfig(
            num_classes=80,
            score_thresh=score_thresh,
            nms_thresh=nms_thresh,
            topk=topk,
        )

        for _ in range(max(0, warmup)):
            _ = postprocess_detections(
                proposals_xyxy=proposals,
                cls_logits=cls_logits,
                bbox_deltas=bbox_deltas,
                mask_logits=mask_logits,
                image_hw=(800, 1216),
                cfg=pp_cfg,
            )

        pp_samples = []
        post = None
        for _ in range(max(1, runs)):
            t0 = time.perf_counter()
            post = postprocess_detections(
                proposals_xyxy=proposals,
                cls_logits=cls_logits,
                bbox_deltas=bbox_deltas,
                mask_logits=mask_logits,
                image_hw=(800, 1216),
                cfg=pp_cfg,
            )
            t1 = time.perf_counter()
            pp_samples.append((t1 - t0) * 1000.0)

        lat["python_postprocess"] = self._latency_stats(pp_samples)
        assert post is not None
        return PredictResult(post=post, latencies=lat)


class MaskRCNN:
    """Factory wrapper.

    Instantiate this class with a provider, and it will create the appropriate backend.

    Example:
        m = MaskRCNN(model_dir, provider="CPUExecutionProvider")
        m = MaskRCNN(model_dir, provider="QAicExecutionProvider")
    """

    def __init__(self, model_dir: str, provider: str = "QAicExecutionProvider"):
        provider = provider.strip()
        if provider == "CPUExecutionProvider":
            self._impl: _MaskRcnnPredictor = MaskRCNNOnnxCPU(model_dir, providers=[provider])
        elif provider == "QAicExecutionProvider":
            self._impl = MaskRCNNQaic(model_dir, providers=[provider])
        else:
            raise ValueError(f"Unsupported provider: {provider}")

    def predict(
        self,
        image_path: str,
        *,
        score_thresh: float = 0.05,
        nms_thresh: float = 0.5,
        topk: int = 100,
        warmup: int = 1,
        runs: int = 10,
    ) -> PredictResult:
        return self._impl.predict(
            image_path,
            score_thresh=score_thresh,
            nms_thresh=nms_thresh,
            topk=topk,
            warmup=warmup,
            runs=runs,
        )
