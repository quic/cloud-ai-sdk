from typing import Dict, List, Set

import numpy as np

try:
    import qaicrt
except ImportError:
    import platform
    import sys

    sys.path.append(f"/opt/qti-aic/dev/lib/{platform.machine()}")
    import qaicrt

try:
    from QAicApi_pb2 import aicapi
except ImportError:
    import sys

    sys.path.append("/opt/qti-aic/dev/python")
    import QAicApi_pb2 as aicapi


aic_to_np_dtype_mapping = {
    aicapi.FLOAT_TYPE: np.dtype(np.float32),
    aicapi.FLOAT_16_TYPE: np.dtype(np.float16),
    aicapi.INT8_Q_TYPE: np.dtype(np.int8),
    aicapi.UINT8_Q_TYPE: np.dtype(np.uint8),
    aicapi.INT16_Q_TYPE: np.dtype(np.int16),
    aicapi.INT32_Q_TYPE: np.dtype(np.int32),
    aicapi.INT32_I_TYPE: np.dtype(np.int32),
    aicapi.INT64_I_TYPE: np.dtype(np.int64),
    aicapi.INT8_TYPE: np.dtype(np.int8),
}


class QAICInferenceSession:
    def __init__(
        self,
        qpc_path: str,
        device_ids: List[int] = [0],
        activate: bool = True,
        enable_debug: bool = False,
    ):

        # Load QPC
        devices = qaicrt.QIDList(device_ids)
        self.context = qaicrt.Context(devices)
        if enable_debug:
            assert (
                self.context.setLogLevel(qaicrt.QLogLevel.QL_DEBUG) == qaicrt.QStatus.QS_SUCCESS
            ), "Failed to setLogLevel"
        qpc = qaicrt.Qpc(qpc_path)

        # Load IO Descriptor
        iodesc = aicapi.IoDesc()
        status, iodesc_data = qpc.getIoDescriptor()
        assert status == qaicrt.QStatus.QS_SUCCESS, "Failed to getIoDescriptor"
        iodesc.ParseFromString(iodesc_data)
        self.allowed_shapes = [
            [(aic_to_np_dtype_mapping[x.type].itemsize, list(x.dims)) for x in allowed_shape.shapes]
            for allowed_shape in iodesc.allowed_shapes
        ]
        self.bindings = iodesc.selected_set.bindings
        self.input_spec = {}
        self.output_spec = {}
        for iobinding in self.bindings:
            if iobinding.dir == aicapi.BUFFER_IO_TYPE_INPUT:
                self.input_spec[iobinding.name] = (
                    aic_to_np_dtype_mapping[iobinding.type],
                    tuple(iobinding.dims),
                    iobinding.index,
                )
            elif iobinding.dir == aicapi.BUFFER_IO_TYPE_OUTPUT:
                self.output_spec[iobinding.name] = (
                    aic_to_np_dtype_mapping[iobinding.type],
                    tuple(iobinding.dims),
                    iobinding.index,
                )

        # Create and load Program
        self.program = qaicrt.Program(self.context, qpcObj=qpc)
        self.program.load()
        prog_properties = qaicrt.QAicProgramProperties()
        prog_properties.SubmitRetryTimeoutMs = 600_000
        self.program.initProperties(prog_properties)
        if activate:
            self.activate()

        # Create input qbuffers and buf_dims
        self.qbuffers = [qaicrt.QBuffer(bytes(binding.size)) for binding in self.bindings]
        self.buf_dims = qaicrt.BufferDimensionsVecRef(
            [
                (aic_to_np_dtype_mapping[binding.type].itemsize, list(binding.dims))
                for binding in self.bindings
            ]
        )

    @property
    def input_names(self) -> List[str]:
        return list(self.input_spec.keys())

    @property
    def output_names(self) -> List[str]:
        return list(self.output_spec.keys())

    def activate(self):
        self.program.activate()
        self.execObj = qaicrt.ExecObj(self.context, self.program)

    def deactivate(self):
        del self.execObj
        self.program.deactivate()

    def set_zero_size_io(self, zero_io_indices: Set[int]):
        # Rebuilding qbuffers is necessary to avoid memory alignment errors
        self.qbuffers = [
            qaicrt.QBuffer(bytes(0)) if i in zero_io_indices else x
            for i, x in enumerate(self.qbuffers)
        ]
        for i in zero_io_indices:
            self.buf_dims[i] = (self.buf_dims[i][0], [0])

    def skip_inputs(self, skipped_input_names: Set[str]):
        skipped_indices = set()
        for input_name in skipped_input_names:
            _, _, input_index = self.input_spec.pop(input_name)
            skipped_indices.add(input_index)
        self.set_zero_size_io(skipped_indices)

    def skip_outputs(self, skipped_output_names: Set[str]):
        skipped_indices = set()
        for output_name in skipped_output_names:
            _, _, output_index = self.output_spec.pop(output_name)
            skipped_indices.add(output_index)
        self.set_zero_size_io(skipped_indices)

    def run(self, inputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        # Set inputs
        for input_name, (input_dtype, input_shape, input_index) in self.input_spec.items():
            input_array = inputs[input_name]
            self.qbuffers[input_index] = qaicrt.QBuffer(input_array.tobytes())
            self.buf_dims[input_index] = (input_array.itemsize, input_array.shape)
        assert (
            self.execObj.setData(self.qbuffers, self.buf_dims) == qaicrt.QStatus.QS_SUCCESS
        ), "Failed to setData"

        # Run the model, get outputs
        if self.execObj.run(self.qbuffers) != qaicrt.QStatus.QS_SUCCESS:
            error_message = "Failed to run"

            # Print additional error messages for unmatched dimension error
            if self.allowed_shapes:
                error_message += " (possible due to incorrect shapes)\n\nAllowed shapes:"
                for i, allowed_shape in enumerate(self.allowed_shapes):
                    error_message += f"\n{i}\n"
                    for binding, (elemsize, shape) in zip(self.bindings, allowed_shape):
                        if (
                            binding.name not in self.input_spec
                            and binding.name not in self.output_spec
                        ):
                            continue
                        error_message += f"{binding.name}:\t{elemsize}\t{shape}\n"
                error_message += f"\n\nPassed shapes:\n"
                for binding, (elemsize, shape) in zip(self.bindings, self.buf_dims):
                    if binding.name not in self.input_spec and binding.name not in self.output_spec:
                        continue
                    error_message += f"{binding.name}:\t{elemsize}\t{shape}\n"
            raise ValueError(error_message)

        status, output_qbuffers = self.execObj.getData()
        assert status == qaicrt.QStatus.QS_SUCCESS, "Failed to getData"
        outputs = {}
        for output_name, (output_dtype, output_shape, output_index) in self.output_spec.items():
            outputs[output_name] = np.frombuffer(
                bytes(output_qbuffers[output_index]), output_dtype
            ).reshape(output_shape)

        return outputs
