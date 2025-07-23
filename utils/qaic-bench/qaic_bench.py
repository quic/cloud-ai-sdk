# # Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

from pprint import pprint
import os
import argparse
import subprocess
import json
import pandas as pd
import re

from datetime import datetime
from QEfficient import QEFFAutoModelForCausalLM
from tempfile import NamedTemporaryFile

def scan_devices(device_filter=None):
    if device_filter:
        device_filter = sorted(device_filter)

    print('device_filter {}'.format(device_filter))

    # skuType in qaicrt not accurate, so use qaic-util instead
    # Get all devices
    tmp_file = NamedTemporaryFile(delete=False)
    tmp_file_name = tmp_file.name
    tmp_file.close()

    command = '/opt/qti-aic/tools/qaic-util -q -j {}'.format(tmp_file_name)

    device_info = {'info': []}
    qaic_util = None

    result = subprocess.run(command.split(), stdout=subprocess.DEVNULL)
    if result.returncode == 0:
        with open(tmp_file_name, 'r') as f:
            qaic_util = json.load(f)
            #print(qaic_util)

        devices_available = []
        for info in qaic_util['device_info']:
            device_id = info['qid']
            if not device_filter or (device_filter and device_id in device_filter):
                devices_available.append(device_id)
                attr_list = ['qid', 'dev_status', 'sku_type', 'nsp_total', 'nsp_free']
                attr_info = {}
                for attr in attr_list:
                    attr_info[attr] = info[attr]
                device_info['info'].append(attr_info)

        device_info['device_list'] = devices_available

    os.unlink(tmp_file_name)

    if not qaic_util:
        print('Could not retrieve device information')
        return None

    return device_info

def generate_device_groups(device_info, num_devices):
    device_group = ''

    for device in range(num_devices):
        device_group += '{},'.format(device_info['device_list'][device])
    device_group = device_group[0:-1]

    return device_group

def qefficient_compile(snapshot, num_devices=1, num_cores=16, batch_size=1, context_len=4096):
    model = QEFFAutoModelForCausalLM.from_pretrained(f"{snapshot}", continuous_batching=True)

    qpc_path = model.compile(
            mxint8_kv_cache=True,
            num_cores=num_cores,
            num_devices=num_devices,
            prefill_seq_len=128,
            ctx_len=context_len,
            aic_enable_depth_first=True,
            mxfp6_matmul=True,
            allow_mxint8_mdp_io=True,
            full_batch_size=batch_size
        )

    return qpc_path

class QBenchVLLM:
    def __init__(self, vllm_root, device_info):
        self.vllm_root = vllm_root
        self.device_info = device_info

        self.vllm_env = os.environ.copy()
        self.vllm_env['VLLM_QAIC_MAX_CPU_THREADS'] = '8'

    def benchmark_throughput(self, model, qpc, num_devices=1, batch_size=1, input_len=2048, output_len=2048):
        cwd = os.getcwd()

        device_group = generate_device_groups(self.device_info, num_devices)
        #print(device_group)

        #json_file = os.path.join(cwd, 'device_{}.json'.format(device_group.replace(',', '_')))

        prompts = 2 * batch_size

        arg_data = {
            '--input-len': '{}'.format(input_len),
            '--output-len': '{}'.format(output_len),
            '--max-num-seqs': '{}'.format(batch_size),
            '--max-seq-len-to-capture': '128',
            '--device': 'qaic',
            '--kv-cache-dtype': 'mxint8',
            '--max-model-len': '{}'.format(input_len + output_len),
            '--num-prompts': '{}'.format(prompts),
            '--quantization': 'mxfp6',
            '--model': model,
            '--device-group': device_group,
            '--temperature': '0.0',
            '--seed': '20',
        }

        command = ['python3', 'benchmarks/benchmark_throughput.py']

        for arg in arg_data.keys():
            command.append(arg)
            command.append(arg_data[arg])

        #command = ['cat', os.path.join(cwd, 'sample_run.txt')]
        print(' '.join(command))

        self.vllm_env['VLLM_QAIC_QPC_PATH'] = qpc
        #print(self.vllm_env)
        result = subprocess.run(command, env=self.vllm_env, cwd=self.vllm_root, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

        if result.returncode == 0:
            stats = {}
            print(result.stdout)
            output_token_rate = 0.0

            for line in result.stdout.splitlines():
                line_new = line.strip()
                if "Processed prompts:" in line_new:
                    match = re.match(r'^Processed prompts.+output\: ([\d\.]+) toks.+', line_new)
                    if match:
                        output_token_rate = float(match.group(1))

            stats['E2E'] = output_token_rate
        else:
            print(result.stderr)
            stats = None
        
        return stats

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', help='JSON file with model configurations')
    parser.add_argument('--devices', help='List of comma separated device IDs to use for inferencing')
    parser.add_argument('--compile-only', action='store_true', help='Generate QPCs and skip benchmarking')
    args = parser.parse_args()

    device_list = None

    if args.devices:
        device_list = list(map(int, args.devices.split(',')))

    device_info = scan_devices(device_list)
    pprint(device_info)

    with open(args.config, 'r') as fp:
        bench_configs = json.load(fp)

    vllm_bench = QBenchVLLM(bench_configs['vllm_root'], device_info)

    all_results = {
        'Model': [],
        'Devices': [],
        'PL': [],
        'GL': [],
        'BS': [],
        'vLLM E2E (tok/s)': [],
        'vLLM Normalized E2E (tok/s)': [],
    }

    print('Benchmarking started at {}'.format(datetime.now()))

    if args.compile_only:
        print('Generating QPCs only, skipping benchmarks')

    if device_info:
        num_cores = device_info['info'][0]['nsp_total']
    else:
        num_cores = 16

    for model in bench_configs['models']:
        for config in model['configs']:
            print('Benchmarking {} {}'.format(model['name'], config))

            all_results['Model'].append(model['name'])
            all_results['Devices'].append(config['devices'])
            all_results['PL'].append(config['prompt_len'])
            all_results['GL'].append(config['generation_len'])
            all_results['BS'].append(config['batch_size'])

            if not 'qpc' in config:
                # Prepare QPC model binary
                print('Generating QPC...')
                context_len = config['prompt_len'] + config['generation_len']
                qpc = qefficient_compile(model['model'], 
                                         config['devices'], 
                                         num_cores,
                                         config['batch_size'], 
                                         context_len)
                config['qpc'] = str(qpc)
                print('Updated qpc {} {}'.format(model['name'], config))

            if not os.path.isfile(os.path.join(config['qpc'], 'programqpc.bin')):
                print('QPC not found, skipping')
                continue

            if args.compile_only:
                continue

            stats = vllm_bench.benchmark_throughput(
                model['model'],
                config['qpc'],
                config['devices'],
                config['batch_size'],
                config['prompt_len'],
                config['generation_len'])
            print(stats)

            if stats:
                all_results['vLLM E2E (tok/s)'].append(stats['E2E'])
                all_results['vLLM Normalized E2E (tok/s)'].append(round(stats['E2E'] / config['batch_size'], 2))
            else:
                all_results['vLLM E2E (tok/s)'].append(' ')
                all_results['vLLM Normalized E2E (tok/s)'].append(' ')

    print('Benchmarking ended at {}'.format(datetime.now()))

    if not args.compile_only:
        print(all_results)

        df = pd.DataFrame.from_dict(all_results)
        print(df)
        df.to_csv('results.csv', index=False)

if __name__=="__main__":
    main()
