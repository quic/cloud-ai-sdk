import sys
sys.path.append("/opt/qti-aic/dev/lib/x86_64/")
import numpy as np
from utils import QAIC_Inference

qpcPath = './simple_nn_bin'

qaic = QAIC_Inference(qpcPath)

num_inferences = 2
for i in range(num_inferences):
    # Feeding a randomly generated vector. Size and datatype should match the input.
    input_data = np.random.rand(1, 128).astype(np.float16)  # dtype to be modified based on given model
    output_arr = qaic.infer(input_data, inf_id=i)
    output_data = np.frombuffer(output_arr, dtype=np.float16) # dtype to be modified based on given model
    print(output_data)
