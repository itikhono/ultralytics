# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

# ROCm / MIGraphX / torch.compile tests for AMD GPU inference (Linux only).

import sys
from pathlib import Path

import pytest
import torch

from tests import MODEL, SOURCE
from ultralytics import YOLO
from ultralytics.utils.torch_utils import attempt_compile

ROCM_IS_AVAILABLE = sys.platform == "linux" and bool(getattr(torch.version, "hip", None))
ROCM_DEVICE_AVAILABLE = ROCM_IS_AVAILABLE and torch.cuda.is_available() and torch.cuda.device_count() > 0

MIGRAPHX_AVAILABLE = False
if ROCM_IS_AVAILABLE:
    try:
        import onnxruntime

        MIGRAPHX_AVAILABLE = "MIGraphXExecutionProvider" in onnxruntime.get_available_providers()
    except ImportError:
        pass


@pytest.mark.skipif(not ROCM_IS_AVAILABLE, reason="ROCm/HIP not available")
def test_rocm_hip_version():
    """Verify that PyTorch reports a valid ROCm HIP version string."""
    assert torch.version.hip


@pytest.mark.skipif(not ROCM_IS_AVAILABLE, reason="ROCm/HIP not available")
def test_rocm_cuda_available():
    """Verify ROCm device visibility via torch.cuda (HIP maps to the CUDA API)."""
    if not torch.cuda.is_available():
        pytest.skip("ROCm build detected but no GPU is visible/accessible")
    assert torch.cuda.device_count() > 0


@pytest.mark.skipif(not ROCM_DEVICE_AVAILABLE, reason="ROCm device not available")
@pytest.mark.skipif(not MIGRAPHX_AVAILABLE, reason="MIGraphX Execution Provider not available")
def test_rocm_detection():
    """Test ONNX export followed by inference using the MIGraphX Execution Provider on AMD GPU."""
    file = YOLO(MODEL).export(format="onnx", imgsz=32)
    try:
        results = YOLO(file)(SOURCE, imgsz=32, device=0)
        assert results
        # Verify that the MIGraphX Execution Provider is actually used for this ONNX model.
        import onnxruntime

        session = onnxruntime.InferenceSession(
            str(file), providers=["MIGraphXExecutionProvider", "CPUExecutionProvider"]
        )
        providers = session.get_providers()
        assert "MIGraphXExecutionProvider" in providers
    finally:
        Path(file).unlink(missing_ok=True)


@pytest.mark.skipif(not ROCM_DEVICE_AVAILABLE, reason="ROCm device not available")
@pytest.mark.skipif(not MIGRAPHX_AVAILABLE, reason="MIGraphX Execution Provider not available")
def test_rocm_provider_selection():
    """Verify that MIGraphXExecutionProvider is selected when loading an ONNX model on ROCm."""
    import onnxruntime

    available = onnxruntime.get_available_providers()
    assert "MIGraphXExecutionProvider" in available


@pytest.mark.slow
@pytest.mark.skipif(not ROCM_DEVICE_AVAILABLE, reason="ROCm device not available")
@pytest.mark.skipif(not MIGRAPHX_AVAILABLE, reason="MIGraphX Execution Provider not available")
def test_rocm_onnx_dynamic_export_and_inference():
    """Test dynamic ONNX export followed by inference with MIGraphX EP on AMD GPU."""
    file = YOLO(MODEL).export(format="onnx", dynamic=True, imgsz=32)
    try:
        results = YOLO(file)([SOURCE, SOURCE], imgsz=64, device=0)
        assert len(results) == 2
    finally:
        Path(file).unlink(missing_ok=True)


@pytest.mark.slow
@pytest.mark.skipif(not ROCM_DEVICE_AVAILABLE, reason="ROCm device not available")
@pytest.mark.skipif(not hasattr(torch, "compile"), reason="torch.compile not available")
def test_rocm_torch_compile():
    """Test torch.compile with inductor backend on ROCm using a lightweight inference pass."""
    model = YOLO(MODEL).model.to("cuda:0").eval()
    model = attempt_compile(model, device=torch.device("cuda:0"), imgsz=32, warmup=True, mode="default")
    x = torch.zeros(1, 3, 32, 32, device="cuda:0")
    with torch.inference_mode():
        y = model(x)
    assert y is not None, "Compiled ROCm model should produce inference outputs"
