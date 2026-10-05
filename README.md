# OSNet ReID TensorRT

Run single-image or batched OSNet person re-identification from a serialized
TensorRT engine. Preprocessing uses OpenCV and NumPy; no PIL, PyTorch model,
pretrained weights, or engine files are included.

## Installation

Requires Python 3.9+, an NVIDIA GPU, and a compatible NVIDIA driver, CUDA,
TensorRT, and PyCUDA environment. Use a TensorRT version compatible with the
engine you intend to load. Installing this package does not build an engine
or install the NVIDIA driver/CUDA toolkit. Building PyCUDA from source requires
the CUDA toolkit and a C++ compiler.

Install directly from GitHub (requires Git). Replace `OWNER` with the repository
owner or organization:

```bash
python -m pip install "git+https://github.com/OWNER/person-reid.git"
```

For editable development, run from a local clone of the repository:

```bash
python -m pip install -e .
```

Dependencies are NumPy, headless OpenCV, TensorRT, and PyCUDA. In an environment
where all dependencies are already managed externally (including a compatible
OpenCV installation), use:

```bash
python -m pip install --no-deps "git+https://github.com/OWNER/person-reid.git"
```

The package has not been published to PyPI.

## Usage

```python
import cv2
from osnet_reid_trt import ReIDTRT

reid = ReIDTRT("osnet.engine", batch_size=32)
frames = [cv2.imread("person_a.jpg"), cv2.imread("person_b.jpg")]
features = reid.extract_features(frames)
similarity = reid.get_similarity(features[0], features[1])

# Single image: [1, embedding_dim], or None for a missing/empty image.
feature = reid.extract_feature(frames[0])

# Override the requested batch size for an individual call.
features = reid.extract_features(frames, batch_size=16)
```

`from trt_handler import ReIDTRT` and
`from trt_handler.osnet_reid import ReIDTRT` are also supported.

Inputs are uint8 NumPy images in OpenCV BGR, BGRA, or grayscale format. Images
are converted to RGB, resized to width 128 and height 256 with `INTER_LINEAR`,
scaled to [0, 1], normalized with ImageNet mean/std, and arranged as NCHW.
OpenCV resizing can produce different values from the original PIL preprocessing.

`extract_features` accepts an iterable and returns a list of L2-normalized
float32 NumPy vectors of shape `[embedding_dim]`, preserving input order.
Missing or empty images retain `None` results. `get_similarity` returns cosine
similarity, or `0.0` if either feature is `None`.

## Engine requirements

- One FP16 or FP32 input with shape `[N, 3, 256, 128]` and one FP16 or FP32
  embedding output with shape `[N, embedding_dim]`.
- Linear, device-resident input and output tensors.
- The trained OSNet backbone and BN neck included in the engine.
- For dynamic dimensions, optimization profile zero must support the image size.

Inference uses the TensorRT tensor-name API. Dynamic batches are capped at the
profile maximum. Short batches are padded when required by a fixed batch engine
or a profile minimum; extra outputs are discarded. Only one batch of images is
preprocessed at a time.

Importing the inference handler initializes a CUDA context through
`pycuda.autoinit`. Use an instance on the thread owning that context and serialize
calls on that instance.

## Build a wheel

```bash
python -m pip install build
python -m build
```

The wheel and source distribution are written to `dist/`. An engine is supplied
separately at runtime.

Packaging follows the [setuptools pyproject.toml configuration](https://setuptools.pypa.io/en/latest/userguide/pyproject_config.html).
