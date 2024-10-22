# Cloud AI Playground Notebook setup

## Python Setup
```
# Setup venv
python3.9 -m venv imagine_env
source imagine_env/bin/activate

# Download and install Qualcomm Imagine Python library
wget --no-check-certificate https://cloudai.cirrascale.com/sdk/_downloads/c0b3ae2b51d9a73b1e80fc6b0405c856/imagine_sdk-0.4.0-py3-none-any.whl

pip3 install pip -U
pip3 install imagine_sdk-0.4.0-py3-none-any.whl

pip3 install notebook
```

## Launch Notebook
```
jupyter notebook --no-browser --ip 0.0.0.0 --port 8080
```



