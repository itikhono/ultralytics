# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import os
from pathlib import Path

import pytest
import torch

from tests import MODEL, ROCM_DEVICE_COUNT, ROCM_IS_AVAILABLE, SOURCE
from ultralytics import YOLO
from ultralytics.utils.torch_utils import attempt_compile

# Build a DEVICES list analogous to test_cuda.py
DEVICES = list(range(ROCM_DEVICE_COUNT)) if ROCM_IS_AVAILABLE else []

# Check MIGraphX provider availability (analogous to test_cuda.py checking for TensorRT locally)
MIGRAPHX_AVAILABLE = False
if ROCM_IS_AVAILABLE:
    try:
        import onnxruntime

        MIGRAPHX_AVAILABLE = "MIGraphXExecutionProvider" in onnxruntime.get_available_providers()
    except ImportError:
        pass


def _save_and_restore_env(func):
    """Decorator that saves/restores CUDA_VISIBLE_DEVICES around a test to prevent cross-test interference."""

    def wrapper(*args, **kwargs):
        original = os.environ.get("CUDA_VISIBLE_DEVICES")
        try:
            return func(*args, **kwargs)
        finally:
            if original is None:
                os.environ.pop("CUDA_VISIBLE_DEVICES", None)
            else:
                os.environ["CUDA_VISIBLE_DEVICES"] = original

    wrapper.__name__ = func.__name__
    wrapper.__doc__ = func.__doc__
    return wrapper


def test_checks():
    """Validate ROCm HIP settings against torch CUDA functions."""
    if ROCM_IS_AVAILABLE:
        assert torch.version.hip
        assert torch.cuda.is_available()
        assert torch.cuda.device_count() == ROCM_DEVICE_COUNT


@_save_and_restore_env
@pytest.mark.skipif(not DEVICES, reason="No ROCm devices available")
@pytest.mark.skipif(not MIGRAPHX_AVAILABLE, reason="MIGraphX Execution Provider not available")
def test_rocm_detection():
    """Test ONNX export followed by inference using the MIGraphX Execution Provider on AMD GPU."""
    file = YOLO(MODEL).export(format="onnx", imgsz=32)
    model = YOLO(file)
    results = model(SOURCE, imgsz=32, device=DEVICES[0])
    assert results
    assert "MIGraphXExecutionProvider" in model.predictor.model.session.get_providers()
    Path(file).unlink()


@pytest.mark.skipif(not DEVICES, reason="No ROCm devices available")
@pytest.mark.skipif(not MIGRAPHX_AVAILABLE, reason="MIGraphX Execution Provider not available")
def test_rocm_provider_selection():
    """Verify that MIGraphXExecutionProvider is available in ONNX Runtime on ROCm."""
    import onnxruntime

    available = onnxruntime.get_available_providers()
    assert "MIGraphXExecutionProvider" in available


@_save_and_restore_env
@pytest.mark.skipif(not ROCM_IS_AVAILABLE, reason="ROCm/HIP not available")
def test_rocm_cpu_fallback():
    """Test ONNX export and inference on CPU when running on a ROCm system."""
    file = YOLO(MODEL).export(format="onnx", imgsz=32)
    model = YOLO(file)
    results = model(SOURCE, imgsz=32, device="cpu")
    assert results
    assert "CPUExecutionProvider" in model.predictor.model.session.get_providers()
    Path(file).unlink()


@_save_and_restore_env
@pytest.mark.slow
@pytest.mark.skipif(not DEVICES, reason="No ROCm devices available")
@pytest.mark.skipif(not MIGRAPHX_AVAILABLE, reason="MIGraphX Execution Provider not available")
def test_rocm_onnx_dynamic_export_and_inference():
    """Test dynamic ONNX export followed by inference with MIGraphX EP on AMD GPU."""
    file = YOLO(MODEL).export(format="onnx", dynamic=True, imgsz=32)
    results = YOLO(file)([SOURCE, SOURCE], imgsz=64, device=DEVICES[0])
    assert len(results) == 2
    Path(file).unlink()


@_save_and_restore_env
@pytest.mark.slow
@pytest.mark.skipif(not DEVICES, reason="No ROCm devices available")
@pytest.mark.skipif(not hasattr(torch, "compile"), reason="torch.compile not available")
def test_rocm_torch_compile():
    """Test torch.compile with inductor backend on ROCm using a lightweight inference pass."""
    device = torch.device(f"cuda:{DEVICES[0]}")
    model = YOLO(MODEL).model.to(device).eval()
    model = attempt_compile(model, device=device, imgsz=32, warmup=True, mode="default")
    x = torch.zeros(1, 3, 32, 32, device=device)
    with torch.inference_mode():
        y = model(x)
    assert y is not None


@_save_and_restore_env
@pytest.mark.skipif(not DEVICES, reason="No ROCm devices available")
def test_rocm_pytorch_predict():
    """Test PyTorch .pt model inference on ROCm GPU device."""
    model = YOLO(MODEL)
    results = model.predict(SOURCE, device=DEVICES[0], imgsz=32)
    assert results and len(results[0].boxes) >= 0


@_save_and_restore_env
@pytest.mark.skipif(not DEVICES, reason="No ROCm devices available")
def test_rocm_pytorch_predict_batch():
    """Test PyTorch .pt model batch inference on ROCm GPU device."""
    model = YOLO(MODEL)
    results = model.predict([SOURCE, SOURCE], device=DEVICES[0], imgsz=32)
    assert len(results) == 2


@_save_and_restore_env
@pytest.mark.slow
@pytest.mark.skipif(not DEVICES, reason="No ROCm devices available")
def test_rocm_pytorch_val():
    """Test validation pipeline with PyTorch model on ROCm GPU device."""
    model = YOLO(MODEL)
    metrics = model.val(data="coco8.yaml", device=DEVICES[0], imgsz=32)
    assert metrics is not None


@_save_and_restore_env
@pytest.mark.slow
@pytest.mark.skipif(not DEVICES, reason="No ROCm devices available")
@pytest.mark.skipif(not MIGRAPHX_AVAILABLE, reason="MIGraphX Execution Provider not available")
def test_rocm_onnx_dynamic_batch_inference():
    """Test dynamic ONNX export with varying batch sizes on MIGraphX EP."""
    file = YOLO(MODEL).export(format="onnx", dynamic=True, imgsz=32)
    model = YOLO(file)
    for batch_size in [1, 2, 4]:
        results = model([SOURCE] * batch_size, imgsz=32, device=DEVICES[0])
        assert len(results) == batch_size
    Path(file).unlink()


@_save_and_restore_env
@pytest.mark.slow
@pytest.mark.skipif(not DEVICES, reason="No ROCm devices available")
@pytest.mark.skipif(not MIGRAPHX_AVAILABLE, reason="MIGraphX Execution Provider not available")
def test_rocm_onnx_dynamic_imgsz_inference():
    """Test dynamic ONNX inference with different image sizes on MIGraphX EP."""
    file = YOLO(MODEL).export(format="onnx", dynamic=True, imgsz=32)
    model = YOLO(file)
    for sz in [32, 64]:
        results = model(SOURCE, imgsz=sz, device=DEVICES[0])
        assert results
    Path(file).unlink()


@_save_and_restore_env
@pytest.mark.skipif(not DEVICES, reason="No ROCm devices available")
@pytest.mark.skipif(not MIGRAPHX_AVAILABLE, reason="MIGraphX Execution Provider not available")
def test_rocm_onnx_export_simplify():
    """Test ONNX export with simplify=True on ROCm (exercises exporter.py ROCm branch)."""
    file = YOLO(MODEL).export(format="onnx", simplify=True, imgsz=32)
    model = YOLO(file)
    results = model(SOURCE, imgsz=32, device=DEVICES[0])
    assert results
    assert "MIGraphXExecutionProvider" in model.predictor.model.session.get_providers()
    Path(file).unlink()


@_save_and_restore_env
@pytest.mark.slow
@pytest.mark.skipif(not DEVICES, reason="No ROCm devices available")
@pytest.mark.skipif(not MIGRAPHX_AVAILABLE, reason="MIGraphX Execution Provider not available")
def test_rocm_onnx_half_precision():
    """Test FP16 ONNX export and inference on ROCm with MIGraphX EP."""
    file = YOLO(MODEL).export(format="onnx", half=True, imgsz=32)
    model = YOLO(file)
    results = model(SOURCE, imgsz=32, device=DEVICES[0])
    assert results
    Path(file).unlink()


@_save_and_restore_env
@pytest.mark.slow
@pytest.mark.skipif(not DEVICES, reason="No ROCm devices available")
@pytest.mark.skipif(not MIGRAPHX_AVAILABLE, reason="MIGraphX Execution Provider not available")
def test_rocm_onnx_io_binding():
    """Test that static ONNX inference on ROCm uses IO binding for zero-copy GPU transfers."""
    file = YOLO(MODEL).export(format="onnx", imgsz=32)
    model = YOLO(file)
    results = model(SOURCE, imgsz=32, device=DEVICES[0])
    assert results

    backend = model.predictor.model
    assert not backend.dynamic, "Static ONNX model should not be marked dynamic"
    assert backend.use_io_binding, "IO binding should be enabled for static ONNX on GPU"
    assert "MIGraphXExecutionProvider" in backend.session.get_providers()

    assert hasattr(backend, "bindings") and backend.bindings, "Output bindings should be pre-allocated"
    for tensor in backend.bindings:
        assert tensor.is_cuda, f"Output tensor should be on GPU, got {tensor.device}"
        assert tensor.device.index == DEVICES[0], f"Output tensor on wrong device: {tensor.device}"

    Path(file).unlink()
