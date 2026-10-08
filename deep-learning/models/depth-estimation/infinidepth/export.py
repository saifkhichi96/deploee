#!/usr/bin/env python3
"""
Export InfiniDepth (RGB-only) depth model to ONNX.

The exported model predicts relative disparity from a single RGB image.
Depth = 1 / max(disparity, 5e-3).

Requires https://github.com/zju3dv/InfiniDepth available in the Python path.

ONNX model inputs:
    image       float32 [1, 3, H, W]  — RGB normalized to [0, 1]
    query_coord float32 [1, N, 2]     — query points in [-1, 1], (y, x) order

ONNX model output:
    disparity   float32 [1, N, 1]     — predicted relative disparity

Typical usage (H=768, W=1024 matches training resolution):

    mamba run -n infinidepth python export_onnx.py \\
        --ckpt checkpoints/depth/infinidepth.ckpt \\
        --output checkpoints/depth/infinidepth.onnx \\
        --input_h 768 --input_w 1024
"""

import argparse
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# ONNX-friendly wrapper
# ---------------------------------------------------------------------------

class OnnxInfiniDepth(nn.Module):
    """
    Thin ONNX-exportable wrapper around the InfiniDepth RGB-only model.

    Strips out:
      - torch.autocast (replaced with plain FP32 ops)
      - RANSAC postprocessing (only triggered when gt_depth is provided)
      - batch-loop chunking (unnecessary at export time; query N is fixed)

    Keeps:
      - DINOv3 intermediate-layer extraction
      - BasicEncoder
      - ImplicitHead feature sampling + MLP decode
    """

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.pretrained = model.pretrained
        self.basic_encoder = model.basic_encoder
        self.depth_implicit_head = model.depth_implicit_head
        # ImageNet normalisation buffers (shape [1, 3, 1, 1])
        self.register_buffer("_mean", model._mean.float().clone())
        self.register_buffer("_std", model._std.float().clone())
        self.patch_size: int = int(model.patch_size)
        self.layer_idxs = model.model_config["layer_idxs"]

    def forward(
        self,
        image: torch.Tensor,       # [B, 3, H, W]  float32  [0, 1]
        query_coord: torch.Tensor, # [B, N, 2]     float32  [-1, 1]
    ) -> torch.Tensor:             # [B, N, 1]     float32  disparity

        # --- DINOv3 features (no autocast — plain FP32) ---
        x_dino = (image - self._mean) / self._std
        features = self.pretrained.get_intermediate_layers(
            x_dino,
            n=self.layer_idxs,
            return_class_token=True,
        )

        # --- BasicEncoder features ---
        x_basic = 2.0 * image - 1.0
        basic_feat = self.basic_encoder(x_basic)  # [B, 128, H/4, W/4]

        # --- Spatial patch dimensions ---
        patch_h = image.shape[2] // self.patch_size
        patch_w = image.shape[3] // self.patch_size

        # --- Encode DINOv3 last-layer tokens → spatial feature map ---
        feat = self.depth_implicit_head._encode_feat(features, patch_h, patch_w)

        # --- Decode disparity at query coordinates ---
        disparity = self.depth_implicit_head._decode_dpt(feat, basic_feat, query_coord)

        return disparity  # [B, N, 1]


# ---------------------------------------------------------------------------
# Model loading (patches CUDA requirement for CPU-mode export)
# ---------------------------------------------------------------------------

def load_infinidepth(ckpt_path: str, device: torch.device) -> nn.Module:
    """Load InfiniDepth checkpoint, patching the hard CUDA requirement."""
    # Temporarily patch cuda availability so the model init doesn't crash on CPU
    import InfiniDepth.model.model as _m
    _orig_avail = torch.cuda.is_available
    _orig_init = _m._BaseInfiniDepthModel.__init__

    # Patch: skip the "cuda()" call and CUDA check inside __init__
    def _patched_init(self_model, model_path=None, encoder="vitl16"):
        nn.Module.__init__(self_model)
        from InfiniDepth.model.block.config import dinov3_model_configs
        from InfiniDepth.model.block.prompt_models import GeneralPromptModel, SelfAttnPromptModel
        from InfiniDepth.model.block.implicit_decoder import ImplicitHead
        from InfiniDepth.model.block.convolution import BasicEncoder
        import os

        self_model.model_config = dinov3_model_configs[encoder]
        local_dinov3_repo = _m._resolve_local_dinov3_repo()
        self_model.pretrained = torch.hub.load(
            local_dinov3_repo,
            f"dinov3_{encoder}",
            source="local",
            pretrained=False,
        )
        self_model.patch_size = 16
        dim = self_model.pretrained.blocks[0].attn.qkv.in_features
        self_model.basic_encoder = BasicEncoder(input_dim=3, output_dim=128, stride=4)
        self_model.depth_implicit_head = ImplicitHead(
            hidden_dim=dim,
            basic_dim=128,
            fusion_type="concat",
            out_dim=1,
            hidden_list=[1024, 256, 32],
        )
        self_model.register_buffer("_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self_model.register_buffer("_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        self_model._init_variant_modules()

        if model_path is not None:
            if os.path.exists(model_path):
                checkpoint = torch.load(model_path, map_location="cpu")
                self_model.load_state_dict({k[9:]: v for k, v in checkpoint["state_dict"].items()})
            else:
                raise FileNotFoundError(f"Model file {model_path} not found")
        # Skip self.cuda() and CUDA check — we'll move to device ourselves

    _m._BaseInfiniDepthModel.__init__ = _patched_init
    try:
        from InfiniDepth.utils.model_utils import build_model
        model = build_model("InfiniDepth", model_path=ckpt_path)
    finally:
        _m._BaseInfiniDepthModel.__init__ = _orig_init

    model = model.to(device).eval()
    return model


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def export(
    ckpt_path: str,
    output_path: str,
    input_h: int = 768,
    input_w: int = 1024,
    opset: int = 17,
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading checkpoint from {ckpt_path} on {device} ...")
    model = load_infinidepth(ckpt_path, device)

    wrapper = OnnxInfiniDepth(model).to(device).eval()

    # Dense query grid at input resolution (same as upsample_ratio=1)
    N = input_h * input_w
    dummy_image = torch.zeros(1, 3, input_h, input_w, dtype=torch.float32, device=device)
    # query_coord: (y, x) normalized to [-1, 1]
    dummy_queries = torch.zeros(1, N, 2, dtype=torch.float32, device=device)

    print(f"Exporting to {output_path} (opset {opset}) ...")
    print(f"  Input resolution: {input_h} x {input_w}  ({N:,} query points)")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            (dummy_image, dummy_queries),
            output_path,
            input_names=["image", "query_coord"],
            output_names=["disparity"],
            dynamic_axes={
                # Allow arbitrary number of query points so the same model
                # can be queried at different output resolutions.
                "query_coord": {1: "num_queries"},
                "disparity":   {1: "num_queries"},
            },
            opset_version=opset,
            do_constant_folding=True,
            # Force the legacy JIT-trace-based exporter; the dynamo exporter
            # (default in PyTorch ≥2.5) does not support the nested tuple
            # output from DINOv3 get_intermediate_layers out of the box.
            dynamo=False,
        )

    size_mb = os.path.getsize(output_path) / 1024 / 1024
    print(f"Saved  {output_path}  ({size_mb:.1f} MB)")

    # Quick sanity-check with onnxruntime
    try:
        import onnxruntime as ort
        import numpy as np

        print("Verifying with onnxruntime ...")
        sess = ort.InferenceSession(output_path, providers=["CPUExecutionProvider"])
        img_np  = np.random.rand(1, 3, input_h, input_w).astype(np.float32)
        H_out, W_out = input_h // 2, input_w // 2  # half-res query to save time
        ys = ((np.arange(H_out) + 0.5) / H_out) * 2.0 - 1.0
        xs = ((np.arange(W_out) + 0.5) / W_out) * 2.0 - 1.0
        grid_y, grid_x = np.meshgrid(ys, xs, indexing="ij")
        queries = np.stack([grid_y, grid_x], axis=-1).reshape(1, -1, 2).astype(np.float32)
        out = sess.run(["disparity"], {"image": img_np, "query_coord": queries})
        disp = out[0]  # [1, N, 1]
        print(f"  onnxruntime output shape: {disp.shape}  range [{disp.min():.4f}, {disp.max():.4f}]")
        print("Verification passed.")
    except ImportError:
        print("onnxruntime not found — skipping verification.")

    print("Done.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export InfiniDepth to ONNX")
    p.add_argument("--ckpt",    default="checkpoints/depth/infinidepth.ckpt",
                   help="Path to the InfiniDepth checkpoint (.ckpt)")
    p.add_argument("--output",  default="checkpoints/depth/infinidepth.onnx",
                   help="Output ONNX file path")
    p.add_argument("--input_h", type=int, default=768,
                   help="Input image height (must be divisible by 16)")
    p.add_argument("--input_w", type=int, default=1024,
                   help="Input image width (must be divisible by 16)")
    p.add_argument("--opset",   type=int, default=17,
                   help="ONNX opset version")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.input_h % 16 != 0 or args.input_w % 16 != 0:
        sys.exit("input_h and input_w must each be divisible by 16 (DINOv3 patch size).")
    export(
        ckpt_path=args.ckpt,
        output_path=args.output,
        input_h=args.input_h,
        input_w=args.input_w,
        opset=args.opset,
    )
