"""OSNet inference from a serialized TensorRT engine (TensorRT 8.5+).

The engine must expose one linear, device-resident NCHW image input and one
[N, embedding_dim] output, and include the trained OSNet backbone and BN neck.
No PyTorch model or pretrained weights are loaded. Calls on an instance must be
serialized and made on the thread owning the PyCUDA context.
"""

import cv2
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit  # Keep a CUDA context active for allocations and inference.


class ReIDTRT:
    def __init__(self, modelPath, batch_size=64):
        """Load an .engine/.plan file and use optimization profile zero.

        Images are resized to 256x128, as in osnet_reid/inference.py.
        Dynamic batch sizes are limited by the engine profile. Fixed batch
        engines and profiles with a minimum batch greater than one are supported
        by padding short batches and discarding their extra outputs.
        """
        self.batch_size = self._batch_size(batch_size)
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        with open(modelPath, "rb") as model_file:
            self.engine = self.runtime.deserialize_cuda_engine(model_file.read())
        if self.engine is None:
            raise RuntimeError(f"Could not deserialize TensorRT engine: {modelPath}")

        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError("Could not create TensorRT execution context")

        names = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]
        inputs = [name for name in names
                  if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT]
        outputs = [name for name in names
                   if self.engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT]
        if len(inputs) != 1 or len(outputs) != 1:
            raise ValueError("Expected exactly one image input and one embedding output")
        self.input_name, self.output_name = inputs[0], outputs[0]
        for name in names:
            if self.engine.get_tensor_location(name) != trt.TensorLocation.DEVICE:
                raise ValueError(f"Tensor {name!r} must reside on the device")
            if self.engine.get_tensor_format(name) != trt.TensorFormat.LINEAR:
                raise ValueError(f"Tensor {name!r} must use linear storage")

        self.input_dtype = np.dtype(trt.nptype(self.engine.get_tensor_dtype(self.input_name)))
        self.output_dtype = np.dtype(trt.nptype(self.engine.get_tensor_dtype(self.output_name)))
        if self.input_dtype not in (np.dtype(np.float16), np.dtype(np.float32)):
            raise ValueError("Expected an FP16 or FP32 image input")
        if self.output_dtype not in (np.dtype(np.float16), np.dtype(np.float32)):
            raise ValueError("Expected FP16 or FP32 embeddings")

        shape = tuple(self.engine.get_tensor_shape(self.input_name))
        if len(shape) != 4 or any(dim not in (-1, expected)
                                  for dim, expected in zip(shape[1:], (3, 256, 128))):
            raise ValueError(f"Expected input shape [N, 3, 256, 128], got {shape}")
        if -1 in shape:
            minimum, _, maximum = self.engine.get_tensor_profile_shape(self.input_name, 0)
            if any(not low <= size <= high for low, size, high in
                   zip(minimum[1:], (3, 256, 128), maximum[1:])):
                raise ValueError("Profile zero must support input images of shape [3, 256, 128]")
            self.min_batch, self.max_batch = int(minimum[0]), int(maximum[0])
        else:
            self.min_batch = self.max_batch = shape[0]
        if not 1 <= self.min_batch <= self.max_batch:
            raise ValueError("Engine has invalid batch dimensions")

        self.stream = cuda.Stream()
        self._buffers = {}
        self._mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]
        self._std = np.array([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]

    @staticmethod
    def _batch_size(value):
        value = int(value)
        if value < 1:
            raise ValueError("batch_size must be positive")
        return value

    def preprocess(self, frame):
        """Convert a uint8 OpenCV BGR/BGRA/gray NumPy image to RGB CHW."""
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            return None
        if frame.dtype != np.uint8:
            raise ValueError("OpenCV frames must contain uint8 pixels")
        if frame.ndim == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        elif frame.ndim == 3 and frame.shape[2] == 1:
            frame = cv2.cvtColor(frame[:, :, 0], cv2.COLOR_GRAY2RGB)
        elif frame.ndim == 3 and frame.shape[2] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)
        elif frame.ndim == 3 and frame.shape[2] == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        else:
            raise ValueError("Expected a grayscale, BGR, or BGRA image")
        image = cv2.resize(frame, (128, 256), interpolation=cv2.INTER_LINEAR)
        tensor = image.astype(np.float32).transpose(2, 0, 1) / 255.0
        return np.ascontiguousarray((tensor - self._mean) / self._std,
                                    dtype=self.input_dtype)

    def _buffer(self, name, size, dtype):
        buffer = self._buffers.get(name)
        if buffer is None or buffer[0].size < size:
            host = cuda.pagelocked_empty(size, dtype)
            device = cuda.mem_alloc(host.nbytes)
            if buffer is not None:
                buffer[1].free()
            buffer = self._buffers[name] = (host, device)
        return buffer

    def _infer_batch(self, tensors):
        count = len(tensors)
        execution_batch = max(count, self.min_batch)
        shape = (execution_batch, 3, 256, 128)
        if not self.context.set_input_shape(self.input_name, shape):
            raise RuntimeError(f"Engine rejected input shape {shape}")
        output_shape = tuple(self.context.get_tensor_shape(self.output_name))
        if len(output_shape) != 2 or output_shape[0] != execution_batch or output_shape[1] < 1:
            raise ValueError(f"Expected resolved [N, embedding_dim] output, got {output_shape}")

        input_size = int(np.prod(shape))
        output_size = int(np.prod(output_shape))
        host_in, device_in = self._buffer(self.input_name, input_size, self.input_dtype)
        host_out, device_out = self._buffer(self.output_name, output_size, self.output_dtype)
        batch = host_in[:input_size].reshape(shape)
        for index, tensor in enumerate(tensors):
            batch[index] = tensor
        batch[count:] = batch[count - 1]
        for name, device in ((self.input_name, device_in), (self.output_name, device_out)):
            if not self.context.set_tensor_address(name, int(device)):
                raise RuntimeError(f"Could not bind tensor {name!r}")
        try:
            cuda.memcpy_htod_async(device_in, host_in[:input_size], self.stream)
            if not self.context.execute_async_v3(stream_handle=self.stream.handle):
                raise RuntimeError("TensorRT inference failed")
            cuda.memcpy_dtoh_async(host_out[:output_size], device_out, self.stream)
        finally:
            self.stream.synchronize()
        features = host_out[:output_size].reshape(output_shape)[:count].astype(np.float32, copy=True)
        norms = np.linalg.norm(features, axis=1, keepdims=True)
        return features / np.maximum(norms, 1e-12)

    def extract_features(self, frames, batch_size=None):
        """Return a list of normalized float32 NumPy vectors in input order.

        None/empty inputs retain a None result. At most one chunk of images is
        preprocessed at a time; requested batch sizes are capped at the engine
        maximum. The frames argument may be any iterable of images.
        """
        limit = min(self.batch_size if batch_size is None else self._batch_size(batch_size),
                    self.max_batch)
        features, tensors, indexes = [], [], []
        for frame in frames:
            index = len(features)
            features.append(None)
            tensor = self.preprocess(frame)
            if tensor is None:
                continue
            tensors.append(tensor)
            indexes.append(index)
            if len(tensors) == limit:
                for target, feature in zip(indexes, self._infer_batch(tensors)):
                    features[target] = feature
                tensors, indexes = [], []
        if tensors:
            for target, feature in zip(indexes, self._infer_batch(tensors)):
                features[target] = feature
        return features

    def extract_feature(self, frame):
        """Return a single [1, embedding_dim] NumPy feature, or None."""
        feature = self.extract_features([frame])[0]
        return None if feature is None else feature[None, :]

    @staticmethod
    def get_similarity(feat1, feat2):
        """Return cosine similarity for two single embeddings."""
        if feat1 is None or feat2 is None:
            return 0.0
        first = np.asarray(feat1, dtype=np.float32).reshape(-1)
        second = np.asarray(feat2, dtype=np.float32).reshape(-1)
        if first.shape != second.shape:
            raise ValueError("Embeddings must have matching dimensions")
        first = first / max(float(np.linalg.norm(first)), 1e-12)
        second = second / max(float(np.linalg.norm(second)), 1e-12)
        return float(np.clip(np.dot(first, second), -1.0, 1.0))
