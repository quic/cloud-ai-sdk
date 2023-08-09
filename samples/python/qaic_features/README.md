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

Sample output for session.print_metrics()

```bash
Number of inferences utilized for calculation are 999
Minimum latency observed 0.0009578340000000001 s
Maximum latency observed 0.002209001 s
Average latency / inference time observed is 0.0012380756316316324 s
P25 / 25% of inferences observed latency less than 0.001095435 s
P50 / 50% of inferences observed latency less than 0.0012522870000000001 s
P75 / 75% of inferences observed latency less than 0.001299786 s
P90 / 90% of inferences observed latency less than 0.002209001 s
P99 / 99% of inferences observed latency less than 0.0016082370000000002 s
Sum of all the inference times 1.2368375560000007 s
Average latency / inference time observed is 0.0012380756316316324 s
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

Sample output for session.print_profile_data()

```bash
|  File-Line-Function  | |  num calls  | |  func time  | |  tot time  |

('~', 0, "<method 'astype' of 'numpy.ndarray' objects>") 1 0.000149101 0.000149101

('~', 0, '<built-in method numpy.empty>') 1 2.38e-06 2.38e-06

('~', 0, '<built-in method numpy.frombuffer>') 1 4.22e-06 4.22e-06
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
