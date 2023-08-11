import os
import sys
import numpy as np


input_shape = [4,3,300,300] 
number_of_inputs = 1
save_as_raw = True
raw_path = "inputFiles/"
os.makedirs(raw_path,exist_ok=True)
input_file = open(f"input_list.txt","w+")


for i in range(number_of_inputs):
    x = np.random.random(input_shape).astype(np.float32)
    print("one random input generated")
    if save_as_raw:
        path_to_save = f'{raw_path}/input_raw_{i}.raw'
        x.tofile(path_to_save)
        input_file.write(f"{path_to_save}\n")
        input_file.flush()


input_file.close()

