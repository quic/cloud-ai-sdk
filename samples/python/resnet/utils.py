import sys
sys.path.append("/opt/qti-aic/dev/lib/x86_64/")
import qaicrt
import numpy as np
import os

class QAIC_Inference:
    def __init__(self,compiled_net_bin_path,qid=None,num_activations=1,set_size=10):
        self.compiled_net_bin_path = compiled_net_bin_path
        self.set_size = set_size
        self.num_activations = num_activations

        # set dev_list
        if qid is None:
            self.status, self.dev_list = qaicrt.Util().getDeviceIds()
        else:
            self.dev_list = qaicrt.QIDList()
            self.dev_list.append(qid)

        print(f'{self.dev_list} device set. Status : {self.status}')
        
        # Create Context
        self.context = qaicrt.Context(self.dev_list)
        self.context.setLogLevel(qaicrt.QLogLevel.QL_WARN) #QL_WARN, QL_INFO, QL_DEBUG

        # Create QPC [Qualcomm Program Container]
        self.qpc = qaicrt.Qpc(compiled_net_bin_path)

        # Create inference_set
        self.inference_set = qaicrt.InferenceSet(self.context, self.qpc, self.dev_list[0], self.set_size, self.num_activations)
        print(f'Inference_set with set_size = {self.set_size} and num_activations = {self.num_activations} on device ID : {self.dev_list[0]}')

        # Create buf mappings
        self.buf_mappings = self.qpc.getBufferMappings()
        self.input_sizes, self.output_sizes = [], []
        self.get_io_size()

    def get_io_size(self):
        opstats_output = 'aiccyclecounts'
        for m in self.buf_mappings:
            if m.ioType == qaicrt.BufferIoTypeEnum.BUFFER_IO_TYPE_OUTPUT and m.bufferName != opstats_output:
                self.output_sizes.append(m.size)
            elif m.ioType == qaicrt.BufferIoTypeEnum.BUFFER_IO_TYPE_INPUT:
                self.input_sizes.append(m.size)

    def get_output_data(self, buffer):
        for m in self.buf_mappings:
            if m.ioType == qaicrt.BufferIoTypeEnum.BUFFER_IO_TYPE_OUTPUT and m.bufferName != 'aiccyclecounts':
                output_buf = buffer[m.index]
                output_arr = np.frombuffer(output_buf, dtype='uint8')
                return output_arr

    def prepare_buf(self, input_data):
        qbuf_list = []
        qbuf_list.append(qaicrt.QBuffer(input_data))

        for size in self.output_sizes:
            np_arr = np.empty(shape=(size,), dtype='uint8')
            qbuf_list.append(qaicrt.QBuffer(np_arr))
        return qbuf_list 
    
    def inference_complete_task(self, inf_id):
        status, inf_handle = self.inference_set.getCompletedId(inf_id) 
        # print(f'Inference {inf_id} status {status}')
        status, buffer = inf_handle.execObj().getData()
        # print(f'Inference execObj {inf_id} status {status}')
        output_arr = self.get_output_data(buffer)
        self.inference_set.putCompleted(inf_handle)
        return output_arr

    def infer(self, input_data, inf_id):
        qbuf_list = self.prepare_buf(input_data)
        _, inf_handle = self.inference_set.getAvailable()
        inf_handle.setBuffers(qbuf_list)
        self.inference_set.submit(inf_handle, inf_id)
        return self.inference_complete_task(inf_id)
    

def generate_bin(onnx_path, batch_size=1, aic_cores=1, precision='fp16'):
    filename, extension = os.path.splitext(onnx_path)
    onnx_folder = os.path.dirname(onnx_path)
    qpc_bin = onnx_folder+filename+'_bin'

    cmd = f'/opt/qti-aic/exec/qaic-exec -m={onnx_path} -aic-hw -aic-hw-version=2.0 -convert-to-{precision} -batchsize={batch_size} -aic-num-cores={aic_cores}  -aic-binary-dir={qpc_bin}'
    os.system(cmd)
    print(f'Running : {cmd}')

    return qpc_bin

from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image

# Define the data transforms
data_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

class CustomImageDataset(Dataset):
    def __init__(self, image_df, data_dir, transform=None):
        self.image_df = image_df
        self.data_dir = data_dir
        self.transform = transform

    def __len__(self):
        return len(self.image_df)

    def __getitem__(self, index):
        image_path = self.data_dir+'/'+self.image_df['filename'][index]
        label = self.image_df['label'][index]
        image = Image.open(image_path).convert('RGB')

        if self.transform is not None:
            image = self.transform(image)

        return image, label
