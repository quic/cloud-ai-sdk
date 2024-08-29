#!/usr/bin/env bash

####################################################################################################
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause-Clear
####################################################################################################

BINARY_FOLDER="./stable_diffusion_AIC/"
LOG_FOLDER="./compile_logs/"

# Note: For AWS DL2q instances, change to 14
NUM_CORES=16

mkdir ${BINARY_FOLDER}
mkdir ${LOG_FOLDER}

########################################################################################################################

# 1. Compile the Text Encoder
rm -rf ${BINARY_FOLDER}text_encoder
/opt/qti-aic/exec/qaic-exec \
    -m=stable_diffusion_v1.5_onnx/text_encoder/model.onnx \
    -aic-hw -aic-hw-version=2.0 \
    -onnx-define-symbol=batch,1 \
    -convert-to-fp16 \
    -onnx-define-symbol=sequence,77 \
    -aic-binary-dir=${BINARY_FOLDER}text_encoder \
    -aic-num-cores=${NUM_CORES} \
    -compile-only \
    2>&1 | ts > ${LOG_FOLDER}text_encoder.log &

########################################################################################################################

# 2. Compile the Safety Checker
rm -rf ${BINARY_FOLDER}safety_checker
/opt/qti-aic/exec/qaic-exec \
    -m=./stable_diffusion_v1.5_onnx/safety_checker/model.onnx \
    -aic-hw -aic-hw-version=2.0 \
    -onnx-define-symbol=batch,1 \
    -convert-to-fp16 \
    -onnx-define-symbol=channels,3 \
    -onnx-define-symbol=height,512 \
    -onnx-define-symbol=width,512 \
    -onnx-define-symbol=clip_height,224 \
    -onnx-define-symbol=clip_width,224 \
    -aic-binary-dir=${BINARY_FOLDER}safety_checker \
    -compile-only \
    2>&1 | ts > ${LOG_FOLDER}safety_checker.log &

########################################################################################################################

# 3. Compile the VAE Encoder
rm -rf ${BINARY_FOLDER}vae_encoder
/opt/qti-aic/exec/qaic-exec \
    -m=stable_diffusion_v1.5_onnx/vae_encoder/model.onnx \
    -aic-hw -aic-hw-version=2.0 \
    -onnx-define-symbol=batch,1 \
    -convert-to-fp16 \
    -onnx-define-symbol=sequence,77 \
    -onnx-define-symbol=channels,3 \
    -onnx-define-symbol=height,512 \
    -onnx-define-symbol=width,512 \
    -aic-binary-dir=${BINARY_FOLDER}vae_encoder \
    -aic-num-cores=${NUM_CORES} \
    -compile-only \
    2>&1 | ts > ${LOG_FOLDER}vae_encoder.log &

########################################################################################################################

# 4. Compile the VAE Decoder
rm -rf ${BINARY_FOLDER}vae_decoder
/opt/qti-aic/exec/qaic-exec \
    -m=stable_diffusion_v1.5_onnx/vae_decoder/model.onnx \
    -aic-hw -aic-hw-version=2.0 \
    -onnx-define-symbol=batch,1 \
    -stats-batchsize=1 \
    -convert-to-fp16 \
    -onnx-define-symbol=sequence,77 \
    -onnx-define-symbol=channels,4 \
    -onnx-define-symbol=height,64 \
    -onnx-define-symbol=width,64 \
    -aic-binary-dir=${BINARY_FOLDER}vae_decoder \
    -aic-num-cores=${NUM_CORES} \
    -multicast-weights \
    -compile-only \
    -aic-enable-depth-first \
    -aic-depth-first-mem=32 \
    2>&1 | ts > ${LOG_FOLDER}vae_decoder.log &

########################################################################################################################

# 5. Compile UNet
rm -rf ${BINARY_FOLDER}unet
/opt/qti-aic/exec/qaic-exec \
    -m=stable_diffusion_v1.5_onnx/unet/model.onnx \
    -aic-hw -aic-hw-version=2.0 \
    -onnx-define-symbol=batch,2 \
    -stats-batchsize=2 \
    -convert-to-fp16 \
    -onnx-define-symbol=sequence,77 \
    -onnx-define-symbol=channels,4 \
    -onnx-define-symbol=height,64 \
    -onnx-define-symbol=width,64 \
    -aic-binary-dir=${BINARY_FOLDER}unet \
    -aic-num-cores=${NUM_CORES} \
    -mos=2 \
    -ols=1 \
    -compile-only \
    2>&1 | ts > ${LOG_FOLDER}unet.log &

echo Waiting for qaic-exec processes to finish ...
wait
