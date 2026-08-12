"""Convert the official WASB basketball checkpoint to portable TorchScript.

The upstream checkpoint contains only an HRNet state dict. This one-time setup
script loads the architecture from an official WASB-SBDT checkout, verifies the
published checkpoint, and exports the tensor-only forward pass used by
CourtVision. The generated model files live under ``backend/models`` and are
ignored by git.
"""

import argparse
import hashlib
import sys
from pathlib import Path

import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / "tmp" / "WASB-SBDT"
DEFAULT_CHECKPOINT = ROOT / "backend" / "models" / "wasb_basketball.pt"
DEFAULT_OUTPUT = (
    ROOT / "backend" / "models" / "wasb_basketball_torchscript.pt"
)
OFFICIAL_CHECKPOINT_SHA256 = (
    "8d1ba9870d0a6ab37b06ab82bed593c6c09133e713810bac475d0c000bb7e948"
)


class _AttrDict(dict):
    __getattr__ = dict.__getitem__


def _attr_dict(value):
    if isinstance(value, dict):
        return _AttrDict({key: _attr_dict(item) for key, item in value.items()})
    if isinstance(value, list):
        return [_attr_dict(item) for item in value]
    return value


class _TensorOutput(torch.nn.Module):
    """Replace the upstream integer-keyed output dict with one tensor."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, inputs):
        return self.model(inputs)[0]


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def convert(source_dir, checkpoint_path, output_path):
    source_dir = source_dir.resolve()
    checkpoint_path = checkpoint_path.resolve()
    output_path = output_path.resolve()
    source_root = source_dir / "src"
    config_path = source_root / "configs" / "model" / "wasb.yaml"
    hrnet_path = source_root / "models" / "hrnet.py"
    for required in (config_path, hrnet_path, checkpoint_path):
        if not required.is_file():
            raise FileNotFoundError(required)

    checkpoint_sha256 = _sha256(checkpoint_path)
    if checkpoint_sha256 != OFFICIAL_CHECKPOINT_SHA256:
        raise ValueError(
            "Unexpected WASB checkpoint SHA256: "
            f"{checkpoint_sha256}; expected {OFFICIAL_CHECKPOINT_SHA256}"
        )

    sys.path.insert(0, str(source_root))
    try:
        from models.hrnet import HRNet
    finally:
        sys.path.pop(0)

    config = _attr_dict(yaml.safe_load(config_path.read_text(encoding="utf-8")))
    model = HRNet(config).eval()
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    wrapped = _TensorOutput(model).eval()
    example = torch.zeros(1, 9, 288, 512)
    with torch.inference_mode():
        traced = torch.jit.trace(
            wrapped,
            example,
            check_trace=False,
        )
        expected = wrapped(example)
        actual = traced(example)
    torch.testing.assert_close(actual, expected)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    traced.save(str(output_path))
    return {
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha256,
        "output": str(output_path),
        "output_sha256": _sha256(output_path),
        "output_shape": list(actual.shape),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Export the official WASB basketball HRNet as TorchScript."
    )
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(convert(args.source_dir, args.checkpoint, args.output))


if __name__ == "__main__":
    main()
