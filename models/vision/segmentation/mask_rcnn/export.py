####################################################################################################
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
####################################################################################################

import argparse
import os
import shutil
import subprocess
import sys
from typing import Optional
from pathlib import Path
from importlib import resources
import models.vision.segmentation.mask_rcnn as mrcnn_pkg

import onnxruntime as ort

import qaic


def run(cmd: list[str], cwd: str | None = None) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=cwd, check=True)


def export_onnx(work_dir: Path, args: argparse.Namespace) -> None:
    """Run Detectron2-based ONNX export pipeline."""
    # 1. Clone detectron2
    d2_dir = work_dir / "detectron2"
    if d2_dir.exists():
        print(f"[export] Using existing detectron2 at {d2_dir}")
    else:
        run(["git", "clone", args.detectron2_repo, str(d2_dir)])
        run(["git", "checkout", args.detectron2_commit], cwd=str(d2_dir))
        run(["pip3", "install", "-e", ".", "--no-build-isolation"], cwd=str(d2_dir))

    # 2. Download COCO val2017 images
    val_zip = work_dir / "val2017.zip"
    if not val_zip.exists():
        run(["wget", "http://images.cocodataset.org/zips/val2017.zip", "-O", str(val_zip)])

    d2_coco_dir = d2_dir / "datasets" / "coco"
    val_dir = d2_coco_dir / "val2017"
    if val_dir.exists():
        run(["rm", "-rf", str(val_dir)])
    run(["unzip", str(val_zip), "-d", str(d2_coco_dir)])

    # 3. Download a test image
    test_image = work_dir / "000000000139.jpg"
    if not test_image.exists():
        run([
            "wget",
            "http://images.cocodataset.org/val2017/000000000139.jpg",
            "-q",
            "-O",
            str(test_image),
        ])

    # 4. Download COCO annotations
    ann_zip = work_dir / "annotations_trainval2017.zip"
    if not ann_zip.exists():
        run([
            "wget",
            "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
            "-O",
            str(ann_zip),
        ])

    ann_dir = d2_coco_dir / "annotations"
    if ann_dir.exists():
        run(["rm", "-rf", str(ann_dir)])
    run(["unzip", str(ann_zip), "-d", str(d2_coco_dir)])

    # 5. Set DETECTRON_ROOT and DETECTRON2_DATASETS, run prepare_for_tests.sh
    env = os.environ.copy()
    env["DETECTRON_ROOT"] = str(work_dir)
    # DETECTRON2_DATASETS should point to the directory that contains the 'coco' folder
    datasets_root = d2_dir / "datasets"
    env["DETECTRON2_DATASETS"] = str(datasets_root) + "/"
    prep_script = d2_dir / "datasets" / "prepare_for_tests.sh"
    if prep_script.exists():
        print(f"[export] Running {prep_script}")
        subprocess.run(["bash", str(prep_script)], cwd=str(work_dir), check=True, env=env)
    else:
        print(f"[export] Warning: {prep_script} not found; skipping prepare_for_tests.sh")

    # 6. Apply Detectron2 patch from installed package data

    try:
        # models.vision.segmentation.mask_rcnn is a package; patch/ is inside it
        with resources.as_file(
            resources.files(mrcnn_pkg) / "patch" / "detectron2_customization.patch"
        ) as p:
            installed_patch_path = p
    except Exception:
        installed_patch_path = None

    if installed_patch_path is not None and installed_patch_path.exists():
        patch_path = installed_patch_path
    else:
        # Fallback for dev environments
        patch_path = work_dir / "patch" / "detectron2_customization.patch"

    if patch_path.exists():
        # Try to apply the patch once; if it fails, assume it is already applied
        # or the tree has diverged, and continue without treating it as fatal.
        print(f"[export] Trying to apply patch {patch_path}")
        result = subprocess.run([
            "git",
            "apply",
            str(patch_path),
        ], cwd=str(d2_dir))
        if result.returncode == 0:
            print(f"[export] Patch applied successfully: {patch_path}")
        else:
            print(
                f"[export] Patch did not apply cleanly (likely already applied or code changed), "
                f"skipping: {patch_path}"
            )
    else:
        print(f"[export] Warning: patch file {patch_path} not found; skipping patch.")

    # 7. Run ONNX export
    deploy_dir = d2_dir / "tools" / "deploy"

    # Ensure detectron2 source is on PYTHONPATH and use the current interpreter
    env["PYTHONPATH"] = str(d2_dir) + os.pathsep + env.get("PYTHONPATH", "")

    cmd = [
        sys.executable,
        "./export_model.py",
        "--config-file",
        os.path.join("../../", args.config_file),
        "--output",
        str(work_dir),
        "--format",
        "onnx",
        "--split-onnx",
        "--fp16",
        "--fp16-min-abs",
        "1e-6",
        "MODEL.WEIGHTS",
        "detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl",
        "MODEL.DEVICE",
        "cpu",
    ]
    print("[export] Running ONNX export via detectron2/tools/deploy/export_model.py")
    subprocess.run(cmd, cwd=str(deploy_dir), check=True, env=env)

    print("[export] ONNX models should now be in:", work_dir)


def export_qaic(work_dir: Path, args: argparse.Namespace) -> None:
    """Compile ONNX models for QAIC and run qaic-runner tests."""
    # Compile ONNX models for QAIC using qaic-compile and smoke-test with qaic-runner
    def qaic_compile(model: str, cores: int, aic_binary_dir: Path, mos: Optional[int] = None, ols: Optional[int] = None, multicast_weights: Optional[bool] = None, dfs: Optional[bool] = None, quantize: Optional[bool] = None, node_precision_info: Optional[Path] = None) -> None:
        if aic_binary_dir.exists():
            run(["rm", "-rf", str(aic_binary_dir)])
        cmd = [
            "/opt/qti-aic/exec/qaic-compile",
            f"-m={model}",
            "-aic-hw",
            "-aic-hw-version=ai100",
            f"-aic-num-cores={cores}",
            f"-aic-binary-dir={aic_binary_dir}",
            "-compile-only",
        ]
        if mos:
            cmd.append(f"-mos={mos}")
        if ols:
            cmd.append(f"-ols={ols}")
        if multicast_weights:
            cmd.append(f"-multicast-weights")
        if dfs:
            cmd.append(f"-aic-enable-depth-first")
        if quantize:
            cmd.extend([
                "-convert-to-quantize",
                "-quantization-schema-activations=symmetric_with_uint8",
                "-quantization-schema-constants=symmetric_with_uint8",
                "-quantization-precision=Int8",
                "-use-random-input-data=gaussian",
            ])
        else:
            cmd.append("-convert-to-fp16")
        if node_precision_info:
            cmd.append(f"-node-precision-info={node_precision_info}")
        run(cmd, cwd=str(work_dir))

    print("[export] Compiling ONNX models for QAIC...")
    # Run backbone and ROI-heads compiles sequentially (can be parallelized if desired)

    backbone_kwargs = {
        "model": "model_backbone_fpn.onnx",
        "cores": min(12, args.max_cores),
        "mos": 1,
        "ols": 4,
        "multicast_weights": True,
        "aic_binary_dir": work_dir / "aic_model_backbone_fpn",
    }
    if args.quantize:
        backbone_kwargs["quantize"] = True
        backbone_kwargs["node_precision_info"] = Path("./fp16_nodes_backbone_fpn.yaml")
    qaic_compile(**backbone_kwargs)

    qaic_compile("model_roi_heads.onnx", cores=min(4, args.max_cores), aic_binary_dir=work_dir / "aic_model_roi_heads")

    print("[export] Running qaic-runner tests...")
    run(["/opt/qti-aic/exec/qaic-runner", "-t", "./aic_model_backbone_fpn"], cwd=str(work_dir))
    run(["/opt/qti-aic/exec/qaic-runner", "-t", "./aic_model_roi_heads"], cwd=str(work_dir))

    print("[export] QAIC binaries and test runs complete.")

class TritonModelConfig:
    def __init__(self, model_path: Path, name: str, backend: str, max_batch_size: int = 0, count: int = 1, model_version: int = 1, inputs: dict = None, outputs: dict = None, parameters: dict = None):
        self.model_path = model_path
        self.name = name
        self.backend = backend
        self.max_batch_size = max_batch_size
        self.count = count
        self.model_version = model_version
        self.input = inputs
        self.outputs = outputs
        self.parameters = parameters

        # Store shapes and data types for Triton config
        # input_tensors / output_tensors keyed by tensor name
        #   { name: {"shape": [...], "dtype": "TYPE_*" } }
        self.input_tensors: dict[str, dict[str, object]] = {}
        self.output_tensors: dict[str, dict[str, object]] = {}

        if backend == "onnxruntime":
            self._init_from_onnx(model_path)
        elif backend == "qaic":
            self._init_from_qaic(model_path)
        elif backend == "python":
            self._init_from_python(model_path)
        else:
            raise ValueError(f"Unsupported Triton backend: {self.backend}")

    def _init_from_onnx(self, model_path: Path) -> None:
        # Use onnxruntime to read model I/O shapes and data types
        sess = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])

        # Map ONNXRuntime type strings to Triton TYPE_* strings
        type_map = {
            "tensor(float)": "TYPE_FP32",
            "tensor(float16)": "TYPE_FP16",
            "tensor(double)": "TYPE_FP64",
            "tensor(int8)": "TYPE_INT8",
            "tensor(int16)": "TYPE_INT16",
            "tensor(int32)": "TYPE_INT32",
            "tensor(int64)": "TYPE_INT64",
            "tensor(uint8)": "TYPE_UINT8",
            "tensor(uint16)": "TYPE_UINT16",
            "tensor(uint32)": "TYPE_UINT32",
            "tensor(uint64)": "TYPE_UINT64",
            "tensor(bool)": "TYPE_BOOL",
        }

        for inp in sess.get_inputs():
            # Convert ORT shape (may contain None) into a Triton-friendly representation
            shape = [d if isinstance(d, int) else -1 for d in inp.shape]
            dtype = type_map.get(inp.type, "TYPE_FP32")
            self.input_tensors[inp.name] = {"shape": shape, "dtype": dtype}

        for out in sess.get_outputs():
            shape = [d if isinstance(d, int) else -1 for d in out.shape]
            dtype = type_map.get(out.type, "TYPE_FP32")
            self.output_tensors[out.name] = {"shape": shape, "dtype": dtype}

    def _init_from_qaic(self, model_path: Path) -> None:
        """Initialize input/output tensors from a QAIC QPC using qaic.Session APIs."""
        sess = qaic.Session(str(model_path))

        input_desc = getattr(sess, "model_input_shape_dict", {})
        output_desc = getattr(sess, "model_output_shape_dict", {})

        def map_qaic_dtype(qaic_dtype) -> str:
            mapping = {
                "float32": "TYPE_FP32",
                "float16": "TYPE_FP16",
                "bfloat16": "TYPE_BF16",
                "int8": "TYPE_INT8",
                "uint8": "TYPE_UINT8",
                "int16": "TYPE_INT16",
                "int32": "TYPE_INT32",
                "int64": "TYPE_INT64",
                "uint16": "TYPE_UINT16",
                "uint32": "TYPE_UINT32",
                "uint64": "TYPE_UINT64",
                "bool": "TYPE_BOOL",
            }
            return mapping.get(str(qaic_dtype), "TYPE_FP32")

        for name, (shape, dtype) in input_desc.items():
            try:
                shape_list = [int(d) for d in shape]
            except TypeError:
                shape_list = list(shape)
            self.input_tensors[name] = {
                "shape": shape_list,
                "dtype": map_qaic_dtype(dtype),
            }

        for name, (shape, dtype) in output_desc.items():
            try:
                shape_list = [int(d) for d in shape]
            except TypeError:
                shape_list = list(shape)
            self.output_tensors[name] = {
                "shape": shape_list,
                "dtype": map_qaic_dtype(dtype),
            }

    def _init_from_python(self, model_path: Path) -> None:
        """Initialize input/output tensors for Triton 'python' backend.

        Uses the `inputs` and `outputs` metadata passed into __init__.
        Format examples:
            inputs = [
                {"name": "cls_logits", "data_type": "TYPE_FP32", "dims": [200, 81]},
                ...
            ]
            outputs = [
                {"name": "cls_results", "data_type": "TYPE_FP32", "dims": [-1, 2]},
                ...
            ]
        """
        if self.input is None or self.outputs is None:
            raise ValueError(
                f"Python backend model '{self.name}' requires 'inputs' and 'outputs' metadata."
            )

        # Populate input_tensors
        for spec in self.input:
            name = spec["name"]
            dtype = spec["data_type"]
            dims = spec["dims"]
            # Normalize dims to a list of ints (allow -1 for dynamic)
            shape = [int(d) for d in dims]
            self.input_tensors[name] = {"shape": shape, "dtype": dtype}

        # Populate output_tensors
        for spec in self.outputs:
            name = spec["name"]
            dtype = spec["data_type"]
            dims = spec["dims"]
            shape = [int(d) for d in dims]
            self.output_tensors[name] = {"shape": shape, "dtype": dtype}

    def tofile(self, model_root: Path):
        # Ensure parent directories exist before writing
        output_file = model_root / "config.pbtxt"
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w+") as f:
            f.write(f"name: \"{self.name}\"\n")
            f.write(f"backend: \"{self.backend}\"\n")
            f.write(f"max_batch_size: {self.max_batch_size}\n")
            f.write("\n")

            # Dump inputs in pbtxt format
            for name, info in self.input_tensors.items():
                shape = info["shape"]
                dtype = info["dtype"]
                f.write("input {\n")
                f.write(f"  name: \"{name}\"\n")
                f.write(f"  data_type: {dtype}\n")
                f.write("  dims: [")
                for i, d in enumerate(shape):

                    sep = ", " if i < len(shape) - 1 else ""
                    f.write(f"{d}{sep}")
                f.write("]\n")
                f.write("}\n")
            f.write("\n")

            # Dump outputs in pbtxt format
            for name, info in self.output_tensors.items():
                shape = info["shape"]
                dtype = info["dtype"]
                f.write("output {\n")
                f.write(f"  name: \"{name}\"\n")
                f.write(f"  data_type: {dtype}\n")
                f.write("  dims: [")
                for i, d in enumerate(shape):

                    sep = ", " if i < len(shape) - 1 else ""
                    f.write(f"{d}{sep}")
                f.write("]\n")
                f.write("}\n")
            f.write("\n")

            if self.parameters:
                for name, value in self.parameters.items():
                    f.write("parameters {\n")
                    f.write(f"  key: \"{name}\"\n")
                    f.write("  value {\n")
                    if isinstance(value, str):
                        f.write(f"    string_value: \"{value}\"\n")
                    else:
                        raise ValueError(f"Value type of {value} not recognized")
                    f.write("  }\n")
                    f.write("}\n")
            f.write("\n")

            f.write("instance_group {\n")
            f.write("  kind: KIND_CPU\n")
            f.write(f"  count: {self.count}\n")
            f.write("}\n")

        # Populate model binary
        output_file = model_root / str(self.model_version)
        if self.backend == "onnxruntime":
             output_file /= "model.onnx"
        elif self.backend == "qaic":
            output_file /= "programqpc.bin"
        elif self.backend == "python":
            output_file /= "model.py"
        output_file.parent.mkdir(parents=True, exist_ok=True)

        try:
            shutil.copyfile(self.model_path, output_file)
        except Exception as e:
            raise RuntimeError(f"Error copying from {self.model_path} to {output_file}") from e

class TritonModelEnsemble:
    def __init__(self, name: str, models: list[TritonModelConfig], max_batch_size: int = 0, model_version: int = 1):
        self.name = name
        self.models = models
        self.max_batch_size = max_batch_size
        self.model_version = model_version

    def tofile(self, model_root: Path) -> None:
        """Write a generic ensemble config.pbtxt chaining models[0] -> ... -> models[N-1].

        Wiring rules:
        - Ensemble inputs = inputs of the first model.
        - Ensemble outputs = outputs of the last model.
        - For each model in order:
          * For each of its inputs, try to bind to an existing tensor with the same name
            (produced by any prior step or from ensemble input).
          * For each of its outputs, create a uniquely named tensor and record it as available.
        """

        output_file = model_root / "config.pbtxt"
        output_file.parent.mkdir(parents=True, exist_ok=True)

        first = self.models[0]
        last = self.models[-1]

        with open(output_file, "w+") as f:
            f.write(f"name: \"{self.name}\"\n")
            f.write("platform: \"ensemble\"\n")
            f.write(f"max_batch_size: {self.max_batch_size}\n")
            f.write("\n")

            # Ensemble inputs from first model
            for name, info in first.input_tensors.items():
                shape = info["shape"]
                dtype = info["dtype"]
                f.write("input {\n")
                f.write(f"  name: \"{name}\"\n")
                f.write(f"  data_type: {dtype}\n")
                f.write("  dims: [")
                for i, d in enumerate(shape):
                    sep = ", " if i < len(shape) - 1 else ""
                    f.write(f"{d}{sep}")
                f.write("]\n")
                f.write("}\n")
            f.write("\n")

            # Ensemble outputs from last model
            for name, info in last.output_tensors.items():
                shape = info["shape"]
                dtype = info["dtype"]
                f.write("output {\n")
                f.write(f"  name: \"{name}\"\n")
                f.write(f"  data_type: {dtype}\n")
                f.write("  dims: [")
                for i, d in enumerate(shape):
                    sep = ", " if i < len(shape) - 1 else ""
                    f.write(f"{d}{sep}")
                f.write("]\n")
                f.write("}\n")
            f.write("\n")

            # Build scheduling with generic chaining
            f.write("ensemble_scheduling {\n")

            # Map of logical tensor name -> ensemble tensor name
            # Initially, ensemble inputs are available with their own names.
            available_tensors: dict[str, str] = {name: name for name in first.input_tensors.keys()}

            for idx, model in enumerate(self.models):
                f.write("  step {\n")
                f.write(f"    model_name: \"{model.name}\"\n")
                f.write("    model_version: -1\n")

                # Inputs: try to bind each input name to an existing ensemble tensor
                for in_name in model.input_tensors.keys():
                    if in_name in available_tensors:
                        src_name = available_tensors[in_name]
                        f.write("    input_map {\n")
                        f.write(f"      key: \"{in_name}\"\n")
                        f.write(f"      value: \"{src_name}\"\n")
                        f.write("    }\n")
                    else:
                        # No tensor with this name produced so far.
                        # For true generic wiring we would need explicit wiring info;
                        # here we skip mapping to avoid creating broken connections.
                        # You may want to raise or log here instead.
                        raise ValueError(f"No available tensor found for input '{in_name}' in model '{model.name}'")

                # Outputs: create new ensemble tensors; for last model, map directly to ensemble outputs
                is_last = (idx == len(self.models) - 1)
                for out_name in model.output_tensors.keys():
                    f.write("    output_map {\n")
                    f.write(f"      key: \"{out_name}\"\n")
                    if is_last:
                        # Last model: direct to ensemble output
                        dest_name = out_name
                    else:
                        dest_name = f"model{idx}_{out_name}"
                    f.write(f"      value: \"{dest_name}\"\n")
                    f.write("    }\n")
                    # Update available_tensors for next steps
                    available_tensors[out_name] = dest_name

                f.write("  }\n")

            f.write("}\n")

        # Create version directory for ensemble (no binary)
        version_dir = model_root / str(self.model_version)
        version_dir.mkdir(parents=True, exist_ok=True)

def export_triton(work_dir: Path, args: argparse.Namespace) -> None:
    """Populate a Triton model repository with ONNXRuntime, QAIC, and an ensemble."""

    models = {
        # ONNX
        "detectron2_maskrcnn_backbone_fpn_onnx": {
            "path": work_dir / "model_backbone_fpn.onnx",
            "backend": "onnxruntime",
            "count": args.triton_model_count,
        },
        "detectron2_maskrcnn_rpn_onnx": {
            "path": work_dir / "model_rpn.onnx",
            "backend": "onnxruntime",
            "count": args.triton_model_count,
        },
        "detectron2_maskrcnn_roi_heads_onnx": {
            "path": work_dir / "model_roi_heads.onnx",
            "backend": "onnxruntime",
            "count": args.triton_model_count,
        },

        # qaic
        "detectron2_maskrcnn_backbone_fpn_qaic": {
            "path": work_dir / "aic_model_backbone_fpn" / "programqpc.bin",
            "backend": "qaic",
            "count": args.triton_model_count,
        },
        "detectron2_maskrcnn_roi_heads_qaic": {
            "path": work_dir / "aic_model_roi_heads" / "programqpc.bin",
            "backend": "qaic",
            "count": args.triton_model_count,
        },

        # postprocess
        "detectron2_maskrcnn_postprocess": {
            "path": work_dir / "triton_postprocess.py",
            "backend": "python",
            "count": args.triton_model_count,
            "inputs": [
                {"name": "cls_logits", "data_type": "TYPE_FP32", "dims": [200, 81]},
                {"name": "bbox_deltas", "data_type": "TYPE_FP32", "dims": [200, 320]},
                {"name": "mask_logits", "data_type": "TYPE_FP32", "dims": [200, 80, 28, 28]},
                {"name": "proposals", "data_type": "TYPE_FP32", "dims": [200, 4]}
            ],
            "outputs": [
                {"name": "cls_results", "data_type": "TYPE_FP32", "dims": [-1, 2]},
                {"name": "cls_counts", "data_type": "TYPE_INT32", "dims": [1]},
                {"name": "boxes", "data_type": "TYPE_FP32", "dims": [-1, 4]}
            ],
            "parameters": {
                "topk": "20"
            },
        },
    }

    model_configs: dict[str, TritonModelConfig] = {}
    for model_name, info in models.items():
        model_path = info["path"]
        backend = info["backend"]
        count = info["count"]
        kwargs = {"model_path": model_path, "name": model_name, "backend": backend, "count": count}
        if info.get("inputs"):
            kwargs["inputs"] = info["inputs"]
        if info.get("outputs"):
            kwargs["outputs"] = info["outputs"]
        if info.get("parameters"):
            kwargs["parameters"] = info["parameters"]
        model_config = TritonModelConfig(**kwargs)

        model_root = Path(args.triton_model_repo) / model_name
        print(f"[triton] Populating {model_root} ({backend})")
        model_config.tofile(model_root)
        model_configs[model_name] = model_config

    # Example ensemble chaining backbone ONNX -> backbone QAIC
    ensemble_name = "detectron2_maskrcnn_ensemble_qaic"
    ensemble_models = [
        model_configs["detectron2_maskrcnn_backbone_fpn_qaic"],
        model_configs["detectron2_maskrcnn_rpn_onnx"],
        model_configs["detectron2_maskrcnn_roi_heads_qaic"],
        model_configs["detectron2_maskrcnn_postprocess"],
    ]
    ensemble_root = Path(args.triton_model_repo) / ensemble_name
    ensemble = TritonModelEnsemble(
        name=ensemble_name,
        models=ensemble_models,
        max_batch_size=0,
    )
    print(f"[triton] Populating ensemble {ensemble_root}")
    ensemble.tofile(ensemble_root)

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export Detectron2 Mask R-CNN model to ONNX and prepare QAIC artifacts.",
    )
    parser.add_argument(
        "--work-dir",
        default=".",
        help="Working directory to clone detectron2 and download COCO data (default: current directory)",
    )
    parser.add_argument(
        "--quantize",
        action="store_true",
        help="If set, export mixed-precision Int8 quantized model binaries instead of pure FP16",
    )
    parser.add_argument(
        "--detectron2-repo",
        default="https://github.com/facebookresearch/detectron2.git",
        help="Git URL for Detectron2 repository",
    )
    parser.add_argument(
        "--detectron2-commit",
        default="48b598b4f61fbb24182a69b521b2a0ba3252b842",
        help="Detectron2 commit hash to checkout",
    )
    parser.add_argument(
        "--config-file",
        default="configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml",
        help="Detectron2 config file (relative to detectron2 root)",
    )
    parser.add_argument(
        "--triton-model-repo",
        type=str,
        default=None,
        help="Generate Triton model repository at the given path",
    )
    parser.add_argument(
        "--triton-model-count",
        type=int,
        default=1,
        help="Number of instances per Triton model",
    )
    parser.add_argument(
        "--max-cores",
        type=int,
        default=16,
        help="Maximum number of NSP cores to use for model compilation",
    )
    args = parser.parse_args()

    work_dir = Path(args.work_dir).resolve()
    work_dir.mkdir(parents=True, exist_ok=True)

    #if args.triton_model_repo:
    #    export_triton(work_dir, args)
    #    exit(1)

    # First export ONNX models
    export_onnx(work_dir, args)
    # Then compile for QAIC and run qaic-runner tests
    export_qaic(work_dir, args)
    # Then generate Triton model repository
    if args.triton_model_repo:
        export_triton(work_dir, args)




if __name__ == "__main__":
    main()
