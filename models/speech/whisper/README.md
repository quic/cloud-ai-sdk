# Whisper

[Whisper](https://github.com/openai/whisper) is a general-purpose speech recognition model. It is trained on a large dataset of diverse audio and is also a multi-task model that can perform multilingual speech recognition as well as speech translation and language identification.

## Environment and dependencies

```commandline
python3.8 -m venv whisper_env
source whisper_env/bin/activate
pip3 install wheel
pip3 install -r requirements.txt

sudo apt-get install libsndfile1
sudo apt-get install ffmpeg
```

## Model generation

The following command generates encoder and decoder ONNX files in the `output_whisper` folder:
```commandline
python3 generateModel.py --model-name base --output-dir output_whisper
```

**Note** Check here for additional model variants:<br>
https://github.com/openai/whisper#available-models-and-languages


## Model compilation

AIC binaries folder

```commandline
mkdir ./whisper_AIC
```

Whisper encoder

```commandline
rm -rf ./whisper_AIC/whisper-encoder
/opt/qti-aic/exec/qaic-exec -m=./output_whisper/encoder_model.onnx -aic-hw -aic-num-cores=12 -mos=2 -ols=1 -convert-to-fp16 -onnx-define-symbol=batch_size,1 -onnx-define-symbol=feature_size,80 -onnx-define-symbol=encoder_sequence_length,3000 -aic-binary-dir=./whisper_AIC/whisper-encoder -compile-only
```

Whisper decoder

```commandline
rm -rf ./whisper_AIC/whisper-decoder
/opt/qti-aic/exec/qaic-exec -m=./output_whisper/decoder_model.onnx -aic-hw -aic-num-cores=12 -mos=2 -ols=1 -convert-to-fp16 -onnx-define-symbol=batch_size,1 -onnx-define-symbol=encoder_sequence_length,1500 -onnx-define-symbol=decoder_sequence_length,150 -aic-binary-dir=./whisper_AIC/whisper-decoder -compile-only
```

## Model execution

```commandline
sudo ./whisper_env/bin/python3 runModel.py
```
