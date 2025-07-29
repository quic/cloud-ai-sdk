# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

from pprint import pprint
import os
import argparse
import subprocess
import json
import pandas as pd
import re
import time
from pathlib import Path

from datetime import datetime
from QEfficient import QEFFAutoModelForCausalLM
from huggingface_hub import login
from tempfile import NamedTemporaryFile

class QBenchDeviceList():
    def __init__(self):
        self.device_info = None
        self.num_cores = 16

    def scan(self, device_filter=None):
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

        self.num_cores = device_info['info'][0]['nsp_total']
        self.device_info = device_info
        return device_info

    def generate_device_groups(self, num_devices):
        device_group = ''

        for device in range(num_devices):
            device_group += '{},'.format(self.device_info['device_list'][device])
        device_group = device_group[0:-1]

        return device_group

class QBenchModelPreparer:
    def __init__(self, snapshot):
        self.snapshot = snapshot
        self.model_base_name = snapshot.split("/")[-1]

        self.model = QEFFAutoModelForCausalLM.from_pretrained(f"{snapshot}", continuous_batching=True)

        if 'XDG_CACHE_HOME' in os.environ:
            self.onnx_root = Path(os.environ['XDG_CACHE_HOME']) / 'qaic_bench'
        else:
            self.onnx_root = Path('~/.cache/qaic_bench').expanduser()

        self.replicate_kv_heads_script = Path(__file__).resolve().parent / 'replicate_kv_heads.py'
        if not os.path.exists(self.replicate_kv_heads_script):
            self.replicate_kv_heads_script = None

        self.onnx_export = None

    @property
    def model_name(self) -> str:
        return self.snapshot.replace('/', '_')

    def export(self, full_batch_size=None, replicate_kv_heads=False):
        self.onnx_export = None

        if replicate_kv_heads and self.replicate_kv_heads_script:
            kv_heads = 2
            new_kv_heads = self.model.model.config.num_key_value_heads * kv_heads

            export_dir = self.onnx_root / '{}-{}'.format(self.model.model_name, self.model.model_hash)

            onnx_path = f"{self.model_base_name}-{new_kv_heads}kvheads"

            onnx_export = export_dir / onnx_path / '{}_kv.onnx'.format(self.model_name.replace('/', '_'))

            if not os.path.exists(onnx_export):
                print('Replicating KV heads')

                if not os.path.exists(export_dir / onnx_path):
                    os.makedirs(export_dir / onnx_path, exist_ok=True)

                cmd = ['python3', self.replicate_kv_heads_script, '--model_name', self.snapshot, '--prompt', 'Hello, world!', '--repeat', '{}'.format(kv_heads)]
                if full_batch_size:
                    cmd.append('--full_batch_size')
                    cmd.append('{}'.format(full_batch_size))

                print(cmd)

                try:
                    result = subprocess.run(cmd, cwd=export_dir, capture_output=True, text=True, check=True)
                    print(result.stdout)
                except subprocess.CalledProcessError as e:
                    print(f"Command failed with error: {e.stderr}")

            self.onnx_export = onnx_export
        else:
            self.onnx_export = self.model.export()

        print('ONNX export {}'.format(self.onnx_export))
        return self.onnx_export

    def compile(self, num_devices=1, num_cores=16, batch_size=1, context_len=4096):
        qpc_path = self.model.compile(
                onnx_path=self.onnx_export,
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
    def __init__(self, vllm_root, devices):
        self.vllm_root = vllm_root
        self.devices = devices

        self.vllm_env = os.environ.copy()
        self.vllm_env['VLLM_QAIC_MAX_CPU_THREADS'] = '8'

    def benchmark_throughput(self, model, qpc, num_devices=1, batch_size=1, input_len=2048, output_len=2048):
        cwd = os.getcwd()

        device_group = self.devices.generate_device_groups(num_devices)

        prompts = batch_size

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

        print(' '.join(command))

        self.vllm_env['VLLM_QAIC_QPC_PATH'] = qpc
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

def qbench_benchmark(devices, config, compile_only=False):
    device_list = None

    if devices:
        device_list = list(map(int, devices.split(',')))

    qaic_devices = QBenchDeviceList()

    qaic_devices.scan(device_list)
    pprint(qaic_devices.device_info)

    with open(config, 'r') as fp:
        bench_configs = json.load(fp)

    vllm_bench = QBenchVLLM(bench_configs['vllm_root'], qaic_devices)

    all_results = {
        'Model': [],
        'Devices': [],
        'Cores': [],
        'PL': [],
        'GL': [],
        'BS': [],
        'vLLM E2E (tok/s)': [],
        'vLLM Normalized E2E (tok/s)': [],
    }

    if compile_only:
        print('Generating QPCs only, skipping benchmarks')

    for model in bench_configs['models']:
        preparer = None

        for config in model['configs']:
            if not compile_only: print('Benchmarking {} {}'.format(model['name'], config))

            if 'cores' in config:
                num_cores = config['cores']
            else:
                num_cores = qaic_devices.num_cores

            all_results['Model'].append(model['name'])
            all_results['Devices'].append(config['devices'])
            all_results['Cores'].append(num_cores)
            all_results['PL'].append(config['prompt_len'])
            all_results['GL'].append(config['generation_len'])
            all_results['BS'].append(config['batch_size'])

            if not 'qpc' in config:
                if not preparer:
                    preparer = QBenchModelPreparer(model['model'])

                if config['devices'] == 16:
                    replicate_kv_heads = True
                else:
                    replicate_kv_heads = False

                preparer.export(replicate_kv_heads=replicate_kv_heads, full_batch_size=config['batch_size'])

                # Prepare QPC model binary
                print('Generating QPC...')
                context_len = config['prompt_len'] + config['generation_len']
                qpc = preparer.compile(config['devices'],
                                       num_cores,
                                       config['batch_size'],
                                       context_len)
                config['qpc'] = str(qpc)
                print('Updated qpc {} {}'.format(model['name'], config))

            if not os.path.isfile(os.path.join(config['qpc'], 'programqpc.bin')):
                print('QPC not found, skipping')
                continue

            if compile_only:
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

    if compile_only:
        df = pd.DataFrame() # Empty dataframe
    else:
        df = pd.DataFrame.from_dict(all_results)

    return df

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', help='JSON file with model configurations')
    parser.add_argument('--devices', help='List of comma separated device IDs to use for inferencing')
    parser.add_argument('--compile-only', action='store_true', help='Generate QPCs and skip benchmarking')
    parser.add_argument('--hf_token', help='Hugging Face access token')
    args = parser.parse_args()

    if args.hf_token:
        login(args.hf_token)

    print('Benchmarking started at {}'.format(datetime.now()))

    results = qbench_benchmark(args.devices, args.config, args.compile_only)

    print('Benchmarking ended at {}'.format(datetime.now()))

    if not results.empty:
        print(results)
        results.to_csv('results_{}.csv'.format(int(time.time())), index=False)

if __name__=="__main__":
    main()
