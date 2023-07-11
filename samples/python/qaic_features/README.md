# Python High-Level API (qaic) features

qaic_features depicts examples on how one can use different features provided by qaic module along with running inferences.

a) Metrics
After running inferences on AIC100 chip, if you want to get statistics regarding inference times, you can use get_metrics method as follows:

```python
#Create Session with enable_metrics = True
session = qaic.Session(
    model_path,
    options_path=yaml_config_path,
    enable_metrics=True)

#Create input dictionary
input_dict = {'data': np.array()}

#Run Inferences
for i in range(100):
    session.run(input_dict)

#Get Metrics
session.print_metrics()
metrics = session.get_metrics()
```

b) Profiling
To profile your inferences being performed on AIC100 chip and get inference time statistic metrics, you can use following methods:

```python
#Create Session with enable_metrics = True
session = qaic.Session(
    model_path,
    options_path=yaml_config_path,
    enable_profiling=True)

#Create input dictionary
input_dict = {'data': np.array()}

#Run Inferences
for i in range(100):
    session.run(input_dict)

#Get Metrics
session.print_metrics()
metrics = session.get_metrics()
session.print_profile_data(n=5)
```

c) Benchmarking
To run benchmarking for model inferences on AIC100 chip, following method can be used:

```python
#Create Session with enable_metrics = True
session = qaic.Session(
    model_path,
    options_path=yaml_config_path)

#Create input dictionary
input_dict = {'data': np.array()}

# Run Benchmarking
input_dict = {'data': x}
    
inf_completed, inf_rate, inf_time, batch_size = session.run_benchmark(input_dict=input_dict)
```
