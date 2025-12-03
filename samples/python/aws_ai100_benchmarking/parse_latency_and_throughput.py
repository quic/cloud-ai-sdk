##############################################################################
# @@-COPYRIGHT-START-@@
#
# Copyright (c) 2023, Qualcomm Technologies, Inc. All Rights Reserved.
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its contributors
#    may be used to endorse or promote products derived from this software
#    without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# @@-COPYRIGHT-END-@@
##############################################################################


import os
import sys
from glob import glob
import pandas as pd
import numpy as np


def get_metric(series, method):
    '''
    This functions computes the average or percentile for a pandas.Series object
    '''
    if method == 'mean' or method == 'avg':
        return series.mean()
    elif method.endswith('pct'):
        prctile = int(method.replace('pct', ''))/100
        return series.quantile(prctile)
    return None
    
    
def get_latency(latency_logs, latency_method):
    '''
    This function computes the latency from the profiling latency
    text files, using the latency_method specified
    '''
    df = pd.concat([pd.read_csv(filename, skiprows=4)
                    for filename in latency_logs])
    col = df.columns[-3] # Execution Total Time in microseconds
    latency_ms = get_metric(df[col], latency_method)/1000.0
    return latency_ms


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Syntax: python parse_latency_and_throughput.py <latency_method> <model_name> <model_name> <model_name>")
        print("where <latency_method> is 'mean', 'avg', or 'Kpct', where K is a number between 0 to 100")
        print("<model_name> should include full path to the model folder where 'outputFiles' and log files are located")
        sys.exit()

    latency_method = sys.argv[1]
    if (latency_method not in ['mean', 'avg']) and (not latency_method.endswith('pct')):
        raise ValueError(f"Methods supported are mean/avg or <N>pct, received {latency_method}")
    model_names = sys.argv[2:]
    print(model_names)

    # parse the logs and print the latency and throughput
    for config in ['best-throughput', 'balanced', 'best-latency']:

        print("******************************************************************")
        print(f"*** Latency: {config} configurations **************************")
        print("******************************************************************")
        for model in model_names:
            config_folders = glob(f"{model}/outputFiles/fp16*{config}")
            print(f"{model}: Found {len(config_folders)} {config} configurations")
            if len(config_folders) == 0:
                continue
            latency_logs = glob(f"{config_folders[0]}/*latency.txt")
            print(f"Model: {model}: Latency ({latency_method}) = {get_latency(latency_logs, latency_method):.3f} ms")

        print("******************************************************************")
        print(f"*** Throughput: {config} configurations *************************")
        print("******************************************************************")
        for model in model_names:
            log_file = f"{model}/{config}.log"
            if not os.path.exists(log_file):
                print("Model: {model}: {log_file} does not exist")
                continue
            with open(log_file, 'r') as fid:
                throughput = np.double([line.split()[-1]
                                        for line in fid.read().splitlines()
                                        if 'Inf/Sec' in line][-1])
                print(f"Model: {model}: Throughput = {throughput:.3f} inf/sec")
    print("******************************************************************")
