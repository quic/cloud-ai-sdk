Tutorials are Jupyter notebools designed to walk the developer through the Cloud AI inference workflow. The tutorials are split into 2 categories - CV and NLP. Overall, the inference workflow for the CV and NLP are quite similar. The differences are primarily related to the usage of the tools in Cloud AI SDK.   

# Pre-requisites
Install qaic python package <br>
`/opt/qti-aic/dev/python/qaic-env/bin/python3.8 install /opt/qti-aic/dev/lib/x86_64/qaic-0.0.1-py3-none-any.whl`

## Jupyter Notebook Setup 

`/opt/qti-aic/dev/python/qaic-env/bin/pip install ipykernel`

`/opt/qti-aic/dev/python/qaic-env/bin/python -m ipykernel install --user --name qaic-env --display-name "qaic-env"`

`source /opt/qti-aic/dev/python/qaic-env/bin/activate`

Clone the Cloud AI repo and run `jupyter notebook --allow-root --ip 0.0.0.0 --no-browser`



