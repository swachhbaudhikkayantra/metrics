"""
trt_inference.py — TensorRT Inference Engine for Jetson Orin Nano
==================================================================
Builds TensorRT engines from FP32 ONNX models in FP16 or INT8 precision.
Engines are cached to disk so they only build once.

Uses ctypes + libcudart directly (works on all cuda-python versions including 13.x).

Usage:
    session = TRTSession("yolov8/best_fp32.onnx", precision="fp16")
    output  = session.run(["output"], {"images": input_np})
    session.destroy()
"""

import ctypes
import os
import time
import numpy as np
from pathlib import Path

try:
    import tensorrt as trt
except ImportError:
    raise ImportError("TensorRT not found. Install via JetPack or: pip3 install tensorrt")

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

# ── Load libcudart via ctypes ─────────────────────────────────────────────────
_LIB_PATHS = [
    "/usr/local/cuda-12.6/targets/aarch64-linux/lib/libcudart.so.12",
    "/usr/local/cuda/lib64/libcudart.so",
    "libcudart.so",
]
_cudart = None
for _p in _LIB_PATHS:
    try:
        _cudart = ctypes.CDLL(_p, mode=ctypes.RTLD_GLOBAL)
        break
    except OSError:
        continue
if _cudart is None:
    raise RuntimeError("libcudart.so not found — is CUDA installed?")

_cudart.cudaSetDevice.restype = ctypes.c_int
_cudart.cudaSetDevice.argtypes = [ctypes.c_int]
_cudart.cudaMalloc.restype = ctypes.c_int
_cudart.cudaMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
_cudart.cudaFree.restype = ctypes.c_int
_cudart.cudaFree.argtypes = [ctypes.c_void_p]
_cudart.cudaMemcpyAsync.restype = ctypes.c_int
_cudart.cudaMemcpyAsync.argtypes = [
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_void_p
]
_cudart.cudaStreamCreate.restype = ctypes.c_int
_cudart.cudaStreamCreate.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
_cudart.cudaStreamDestroy.restype = ctypes.c_int
_cudart.cudaStreamDestroy.argtypes = [ctypes.c_void_p]
_cudart.cudaStreamSynchronize.restype = ctypes.c_int
_cudart.cudaStreamSynchronize.argtypes = [ctypes.c_void_p]

_MEMCPY_H2D = 1   # cudaMemcpyHostToDevice
_MEMCPY_D2H = 2   # cudaMemcpyDeviceToHost


def _check(ret, op="CUDA"):
    if ret != 0:
        raise RuntimeError(f"{op} error code {ret}")


def _cuda_malloc(nbytes: int) -> ctypes.c_void_p:
    ptr = ctypes.c_void_p()
    _check(_cudart.cudaMalloc(ctypes.byref(ptr), int(nbytes)), "cudaMalloc")
    return ptr


def _cuda_free(ptr: ctypes.c_void_p):
    _cudart.cudaFree(ptr)


def _h2d(host_arr: np.ndarray, dev_ptr: ctypes.c_void_p, stream: ctypes.c_void_p):
    _check(
        _cudart.cudaMemcpyAsync(
            dev_ptr, host_arr.ctypes.data_as(ctypes.c_void_p),
            host_arr.nbytes, _MEMCPY_H2D, stream
        ),
        "cudaMemcpyH2D",
    )


def _d2h(dev_ptr: ctypes.c_void_p, host_arr: np.ndarray, stream: ctypes.c_void_p):
    _check(
        _cudart.cudaMemcpyAsync(
            host_arr.ctypes.data_as(ctypes.c_void_p), dev_ptr,
            host_arr.nbytes, _MEMCPY_D2H, stream
        ),
        "cudaMemcpyD2H",
    )


def _stream_create() -> ctypes.c_void_p:
    stream = ctypes.c_void_p()
    _check(_cudart.cudaStreamCreate(ctypes.byref(stream)), "cudaStreamCreate")
    return stream


# ── TRT numpy dtype map ───────────────────────────────────────────────────────
_TRT_TO_NP = {
    trt.DataType.FLOAT: np.float32,
    trt.DataType.HALF:  np.float16,
    trt.DataType.INT8:  np.int8,
    trt.DataType.INT32: np.int32,
    trt.DataType.BOOL:  np.bool_,
}

# ── Engine builder ────────────────────────────────────────────────────────────

def build_engine(
    onnx_path: str,
    precision: str = "fp32",
    cache_dir: str = None,
    workspace_gb: int = 2,
) -> trt.ICudaEngine:
    """
    Build (or load from cache) a TensorRT engine from a FP32 ONNX model.

    Parameters
    ----------
    onnx_path    : path to the FP32 ONNX file
    precision    : 'fp32' | 'fp16' | 'int8'
    cache_dir    : directory to cache .trt engines (default: same dir as onnx)
    workspace_gb : GPU workspace in GB
    """
    onnx_path = Path(onnx_path)
    cache_dir = Path(cache_dir) if cache_dir else onnx_path.parent
    cache_path = cache_dir / f"{onnx_path.stem}_{precision}.trt"

    # Init CUDA device (must be done before TRT)
    _check(_cudart.cudaSetDevice(0), "cudaSetDevice")

    runtime = trt.Runtime(TRT_LOGGER)

    # ── Load from cache ───────────────────────────────────────────────────────
    if cache_path.exists():
        print(f"  [TRT] Loading cached engine: {cache_path.name}")
        with open(cache_path, "rb") as f:
            engine = runtime.deserialize_cuda_engine(f.read())
        if engine is not None:
            return engine
        print("  [TRT] Cache stale, rebuilding…")
        cache_path.unlink(missing_ok=True)

    # ── Build from ONNX ───────────────────────────────────────────────────────
    print(f"  [TRT] Building {precision.upper()} engine from {onnx_path.name}")
    print(f"        ⏳ First-time build can take 2–8 min — engine cached after this run")
    t0 = time.perf_counter()

    builder = trt.Builder(TRT_LOGGER)
    flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(flags)
    parser = trt.OnnxParser(network, TRT_LOGGER)

    with open(onnx_path, "rb") as f:
        raw = f.read()
    if not parser.parse(raw):
        errors = [str(parser.get_error(i)) for i in range(parser.num_errors)]
        raise RuntimeError("TRT ONNX parse failed:\n" + "\n".join(errors))

    config = builder.create_builder_config()
    config.set_memory_pool_limit(
        trt.MemoryPoolType.WORKSPACE, int(workspace_gb * (1 << 30))
    )

    if precision == "fp16":
        if not builder.platform_has_fast_fp16:
            print("  [TRT] WARNING: FP16 not natively fast here, results will be FP32 speed")
        config.set_flag(trt.BuilderFlag.FP16)

    elif precision == "int8":
        config.set_flag(trt.BuilderFlag.INT8)
        config.set_flag(trt.BuilderFlag.FP16)  # FP16 fallback for layers that can't do INT8
        # PREFER (not OBEY) lets TRT fall back to FP16/FP32 for layers it can't quantize
        # This is important for YOLOv11's segmentation prototype head
        config.set_flag(trt.BuilderFlag.PREFER_PRECISION_CONSTRAINTS)

        # Minimal calibrator using random data (valid for latency benchmarking)
        inp = network.get_input(0)
        inp_shape = tuple(max(int(d), 1) for d in inp.shape)

        class _RandCalib(trt.IInt8MinMaxCalibrator):
            def __init__(self):
                super().__init__()
                self._idx = 0
                self._n = 10
                self._data = np.random.rand(*inp_shape).astype(np.float32)
                self._dev = _cuda_malloc(self._data.nbytes)
                _h2d(self._data, self._dev, None)  # stream=NULL → synchronous

            def get_batch_size(self):
                return inp_shape[0]

            def get_batch(self, names):
                if self._idx >= self._n:
                    return None
                self._idx += 1
                return [int(self._dev.value)]

            def read_calibration_cache(self):
                return None

            def write_calibration_cache(self, cache):
                pass

        config.int8_calibrator = _RandCalib()

    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("TRT build_serialized_network returned None — check GPU memory")


    elapsed = time.perf_counter() - t0
    print(f"  [TRT] ✅ Engine built in {elapsed:.1f}s — saving to {cache_path.name}")

    cache_dir.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "wb") as f:
        f.write(serialized)

    return runtime.deserialize_cuda_engine(serialized)


# ── TRT Inference Session ─────────────────────────────────────────────────────

class TRTSession:
    """
    Drop-in ORT-compatible session for TensorRT inference.
    Provides: get_inputs(), get_outputs(), get_providers(), run()
    """

    def __init__(self, onnx_fp32_path: str, precision: str = "fp16", cache_dir: str = None):
        self.precision = precision
        self.engine = build_engine(onnx_fp32_path, precision, cache_dir)
        self.context = self.engine.create_execution_context()
        self.stream = _stream_create()

        self._input_meta  = []
        self._output_meta = []
        self._dev_ptrs    = {}   # name → device pointer
        self._host_arrs   = {}   # name → host numpy array

        for i in range(self.engine.num_io_tensors):
            name  = self.engine.get_tensor_name(i)
            shape = tuple(max(int(d), 1) for d in self.engine.get_tensor_shape(name))
            dtype = _TRT_TO_NP.get(self.engine.get_tensor_dtype(name), np.float32)
            host  = np.empty(shape, dtype=dtype)
            dev   = _cuda_malloc(host.nbytes)
            self._host_arrs[name] = host
            self._dev_ptrs[name]  = dev
            mode = self.engine.get_tensor_mode(name)
            meta = _TensorMeta(name, list(shape), "tensor(float)")
            if mode == trt.TensorIOMode.INPUT:
                self._input_meta.append(meta)
            else:
                self._output_meta.append(meta)

    def get_inputs(self):
        return self._input_meta

    def get_outputs(self):
        return self._output_meta

    def get_providers(self):
        return [f"TensorrtExecutionProvider ({self.precision.upper()})"]

    def run(self, output_names, input_dict):
        """
        Run inference.  input_dict: {name: np.ndarray}  → list of np.ndarray outputs
        """
        # Upload inputs
        for name, arr in input_dict.items():
            host = self._host_arrs[name]
            np.copyto(host, arr.reshape(host.shape))
            self.context.set_tensor_address(name, int(self._dev_ptrs[name].value))
            _h2d(host, self._dev_ptrs[name], self.stream)

        # Set output addresses
        for name in [m.name for m in self._output_meta]:
            self.context.set_tensor_address(name, int(self._dev_ptrs[name].value))

        # Execute asynchronously
        self.context.execute_async_v3(int(self.stream.value))

        # Download outputs
        results = []
        for m in self._output_meta:
            host = self._host_arrs[m.name]
            _d2h(self._dev_ptrs[m.name], host, self.stream)
        _check(_cudart.cudaStreamSynchronize(self.stream), "cudaStreamSync")

        return [self._host_arrs[m.name].copy() for m in self._output_meta]

    def destroy(self):
        _cudart.cudaStreamDestroy(self.stream)
        for dev in self._dev_ptrs.values():
            _cuda_free(dev)


class _TensorMeta:
    """Mimics onnxruntime NodeArg so benchmark code stays unchanged."""
    def __init__(self, name, shape, type_str):
        self.name  = name
        self.shape = shape
        self.type  = type_str

