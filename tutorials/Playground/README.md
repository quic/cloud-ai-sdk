# Cloud AI Playground Notebook setup

## Python Setup
```
# Setup venv
python3.9 -m venv imagine_env
source imagine_env/bin/activate
pip3 install pip -U

# Install Qualcomm Imagine Python library
pip3 install python-imagine-sdk

# Install dependencies
pip3 install Pillow
pip3 install notebook
pip3 install pandas
```

## Launch Notebook
```
jupyter notebook --no-browser --ip 0.0.0.0 --port 8080
```



