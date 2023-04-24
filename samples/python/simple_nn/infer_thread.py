import threading
import numpy as np
from utils import QAIC_Inference

qpcPath = './simple_nn_bin'
qaic = QAIC_Inference(qpcPath,qid=None,num_activations=10,set_size=1)

num_inferences = 1000
thread_list = []
for i in range(num_inferences):
    # Feeding a randomly generated vector. Size and datatype should match the input.
    input_data = np.random.rand(1, 128).astype(np.float16) #get the type and size from get_io_size?
    inf_complete_thread = threading.Thread(target=qaic.infer, args=(input_data, i))
    thread_list.append(inf_complete_thread)
    inf_complete_thread.start()

for thread in thread_list:
    thread.join()

qaic.inference_set.waitForCompletion()
