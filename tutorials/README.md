Tutorials are Jupyter notebools designed to walk the developer through the Cloud AI inference workflow. The tutorials are split into 2 categories - CV and NLP. Overall, the inference workflow for CV and NLP models are very similar and have been presented for convenience. 

`Model-Onboarding` - This is one of the beginner notebooks. This goes through exporting and preparing the model, compiling the model using a CLI tool and executing inference using CLI tool / Python APIs. 

`Performance-Tuning` - This is another beginner notebook that walks the developer through the key parameters to optimize for best performance (latency and throughput) on Cloud AI platforms. Going through this notebook and 'Performance Tuning' section in the Quick start guide will equip developers with a intuitive understanding of how to use the key parameters to meet inference application KPIs (AI compute resource usage, throughput and latency).   

`Profiler` - This is a intermediate-level notebook that describes system and device level inference profiling capabilities. Developers can use the tools and techniques described in this tutorial to measure application/device level latency and identify system/device bottlenecks. 


### Pre-requisites
Install qaic python package <br>
`/opt/qti-aic/dev/python/qaic-env/bin/python3.8 install /opt/qti-aic/dev/lib/x86_64/qaic-0.0.1-py3-none-any.whl`

### Jupyter Notebook Setup 

`/opt/qti-aic/dev/python/qaic-env/bin/pip install ipykernel`

`/opt/qti-aic/dev/python/qaic-env/bin/python -m ipykernel install --user --name qaic-env --display-name "qaic-env"`

`source /opt/qti-aic/dev/python/qaic-env/bin/activate`

Clone the Cloud AI repo and run `jupyter notebook --allow-root --ip 0.0.0.0 --no-browser`



