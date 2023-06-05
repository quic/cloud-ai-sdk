###############################################################################
# Generate random inputs for BERT base model
###############################################################################

#!/bin/bash
#Input data generation
python -W ignore generateInputs.py --seq_len 128 --batch_size=1
