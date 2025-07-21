## Installation steps

### Create python virtual environment and activate it
```
python3.10 -m venv qeff_env
source qeff_env/bin/activate
pip install --upgrade pip
```

### Clone and install the efficient transformers repo
```
pip install git+https://github.com/quic/efficient-transformers@release/v1.19
```

### After installation of efficient transformers library, install jupyter notebook
```
pip install notebook
```

### Launch Notebook
```
jupyter notebook --no-browser --allow-root
```
