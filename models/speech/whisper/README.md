# Whisper

[Whisper](https://github.com/openai/whisper) is a general-purpose speech recognition model. It is trained on a large dataset of diverse audio and is also a multi-task model that can perform multilingual speech recognition as well as speech translation and language identification.

## Environment and dependencies

```commandline
python3.8 -m venv whisper_env
source whisper_env/bin/activate
pip install -r requirements.txt
```

## Model generation

```commandline
python3 generateModel.py --model-name openai/whisper-base
```

**Note** Similary we can generate based on other variants, please check the variants here.<br>
https://github.com/openai/whisper#available-models-and-languages

## Model compilation

Whisper Model

```commandline
rm -rf ./whisper_AIC
/opt/qti-aic/exec/qaic-exec -m=./whisper-base.onnx -aic-hw -aic-num-cores=12 -mos=2 -ols=1 -convert-to-fp16 -aic-binary-dir=./whisper_AIC -compile-only
```

## Model execution

```commandline
sudo ./whisper_env/bin/python3 runModel.py
```
