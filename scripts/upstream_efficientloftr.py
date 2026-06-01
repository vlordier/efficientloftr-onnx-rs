#!/usr/bin/env python3

from __future__ import annotations

from copy import deepcopy
import math
from pathlib import Path
import sys
import types
from typing import Any, cast

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.einops import rearrange
from kornia.geometry.subpix import dsnt
from kornia.utils import create_meshgrid


def _import_upstream(upstream_root: Path):
    upstream_root = upstream_root.resolve()
    if not (upstream_root / "src" / "loftr").exists():
        raise FileNotFoundError(
            f"upstream repo not found at {upstream_root}; expected src/loftr"
        )
    # Upstream imports `kornia.utils.grid.create_meshgrid`, which moved in newer kornia.
    # Register a tiny shim so upstream code can run without edits.
    if "kornia.utils.grid" not in sys.modules:
        from kornia.utils import create_meshgrid

        shim = types.ModuleType("kornia.utils.grid")
        setattr(shim, "create_meshgrid", create_meshgrid)
        sys.modules["kornia.utils.grid"] = shim

    sys.path.insert(0, str(upstream_root))
    from src.loftr import LoFTR, full_default_cfg, opt_default_cfg, reparameter  # type: ignore[import-not-found]

    return LoFTR, full_default_cfg, opt_default_cfg, reparameter


def load_upstream_matcher(
    upstream_root: Path,
    checkpoint: Path,
    model_type: str = "full",
    export_safe: bool = False,
) -> nn.Module:
    LoFTR, full_default_cfg, opt_default_cfg, reparameter = _import_upstream(upstream_root)

    if model_type == "full":
        cfg = deepcopy(full_default_cfg)
    elif model_type == "opt":
        cfg = deepcopy(opt_default_cfg)
    else:
        raise ValueError(f"unsupported model_type={model_type}; expected full|opt")

    cfg["mp"] = False
    cfg["half"] = False
    cfg["replace_nan"] = False

    matcher = LoFTR(config=cfg)
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    matcher.load_state_dict(state_dict, strict=True)
    matcher = reparameter(matcher)
    _patch_fine_preprocess_for_export(matcher)
    _patch_fine_matching_for_export(matcher, export_safe=export_safe)
    matcher.eval()
    return matcher


def _patch_fine_preprocess_for_export(matcher: nn.Module) -> None:
    fine = cast(Any, matcher).fine_preprocess

    def export_friendly_forward(self, feat_c0, feat_c1, data):
        w = self.W
        stride_raw = data["hw0_f"][0] // data["hw0_c"][0]
        if torch.is_tensor(stride_raw):
            stride = int(stride_raw.item())
        else:
            stride = int(stride_raw)

        data.update({"W": w})

        if data["hw0_i"] == data["hw1_i"]:
            feat_c = rearrange(
                torch.cat([feat_c0, feat_c1], 0),
                "b (h w) c -> b c h w",
                h=data["hw0_c"][0],
            )
            x2 = data["feats_x2"]
            x1 = data["feats_x1"]
            del data["feats_x2"], data["feats_x1"]

            x1 = self.inter_fpn(feat_c, x2, x1, stride)
            feat_f0, feat_f1 = torch.chunk(x1, 2, dim=0)

            feat_f0 = F.unfold(feat_f0, kernel_size=(w, w), stride=stride, padding=0)
            feat_f0 = rearrange(feat_f0, "n (c ww) l -> n l ww c", ww=w**2)
            feat_f1 = F.unfold(feat_f1, kernel_size=(w + 2, w + 2), stride=stride, padding=1)
            feat_f1 = rearrange(feat_f1, "n (c ww) l -> n l ww c", ww=(w + 2) ** 2)

            feat_f0 = feat_f0[data["b_ids"], data["i_ids"]]
            feat_f1 = feat_f1[data["b_ids"], data["j_ids"]]
            return feat_f0, feat_f1

        feat_c0 = rearrange(feat_c0, "b (h w) c -> b c h w", h=data["hw0_c"][0])
        feat_c1 = rearrange(feat_c1, "b (h w) c -> b c h w", h=data["hw1_c"][0])
        x2_0, x2_1 = data["feats_x2_0"], data["feats_x2_1"]
        x1_0, x1_1 = data["feats_x1_0"], data["feats_x1_1"]
        del data["feats_x2_0"], data["feats_x1_0"], data["feats_x2_1"], data["feats_x1_1"]

        feat_f0 = self.inter_fpn(feat_c0, x2_0, x1_0, stride)
        feat_f1 = self.inter_fpn(feat_c1, x2_1, x1_1, stride)

        feat_f0 = F.unfold(feat_f0, kernel_size=(w, w), stride=stride, padding=0)
        feat_f0 = rearrange(feat_f0, "n (c ww) l -> n l ww c", ww=w**2)
        feat_f1 = F.unfold(feat_f1, kernel_size=(w + 2, w + 2), stride=stride, padding=1)
        feat_f1 = rearrange(feat_f1, "n (c ww) l -> n l ww c", ww=(w + 2) ** 2)

        feat_f0 = feat_f0[data["b_ids"], data["i_ids"]]
        feat_f1 = feat_f1[data["b_ids"], data["j_ids"]]
        return feat_f0, feat_f1

    fine.forward = types.MethodType(export_friendly_forward, fine)


def _patch_fine_matching_for_export(matcher: nn.Module, export_safe: bool = False) -> None:
    fine = cast(Any, matcher).fine_matching
    w_fixed = int(cast(Any, matcher).fine_preprocess.W)

    def export_friendly_forward(self, feat_0, feat_1, data):
        m, _, c = feat_0.shape
        w = w_fixed
        ww = w * w
        scale = data["hw0_i"][0] / data["hw0_f"][0]
        self.M, self.W, self.WW, self.C, self.scale = m, w, ww, c, scale

        feat_f0 = feat_0[..., : -self.local_regress_slicedim]
        feat_f1 = feat_1[..., : -self.local_regress_slicedim]
        feat_ff0 = feat_0[..., -self.local_regress_slicedim :]
        feat_ff1 = feat_1[..., -self.local_regress_slicedim :]
        feat_f0 = feat_f0 / c**0.5
        feat_f1 = feat_f1 / c**0.5
        conf_matrix_f = torch.einsum("mlc,mrc->mlr", feat_f0, feat_f1)
        conf_matrix_ff = torch.einsum(
            "mlc,mrc->mlr",
            feat_ff0,
            feat_ff1 / (self.local_regress_slicedim) ** 0.5,
        )

        softmax_matrix_f = F.softmax(conf_matrix_f, 1) * F.softmax(conf_matrix_f, 2)
        softmax_matrix_f = softmax_matrix_f.reshape(m, self.WW, self.W + 2, self.W + 2)
        softmax_matrix_f = softmax_matrix_f[..., 1:-1, 1:-1].reshape(m, self.WW, self.WW)

        conf_flat = softmax_matrix_f.reshape(m, -1)
        mconf, idx = torch.max(conf_flat, dim=-1)
        idx = idx.unsqueeze(-1)
        idx_l = idx // ww
        idx_r = idx % ww

        data.update({"idx_l": idx_l, "idx_r": idx_r, "mconf": mconf})

        grid = create_meshgrid(w, w, False, conf_matrix_ff.device) - w // 2 + 0.5
        grid = grid.reshape(1, -1, 2).expand(m, -1, -1)
        delta_l = torch.gather(grid, 1, idx_l.unsqueeze(-1).expand(-1, -1, 2)).reshape(-1, 2)
        delta_r = torch.gather(grid, 1, idx_r.unsqueeze(-1).expand(-1, -1, 2)).reshape(-1, 2)

        def normalize_scale(scale_like):
            if torch.is_tensor(scale_like):
                t = scale_like.to(device=conf_matrix_ff.device, dtype=delta_l.dtype)
            else:
                t = torch.tensor(scale_like, device=conf_matrix_ff.device, dtype=delta_l.dtype)

            if t.dim() == 0:
                return t.expand(m, 2)
            if t.dim() == 1:
                if t.shape[0] == m:
                    return t.unsqueeze(-1).expand(-1, 2)
                if t.shape[0] == 2:
                    return t.unsqueeze(0).expand(m, -1)
            return t

        if "scale0" in data:
            scale0 = normalize_scale(scale * data["scale0"][data["b_ids"]])
            scale1 = normalize_scale(scale * data["scale1"][data["b_ids"]])
        else:
            scale0 = normalize_scale(scale)
            scale1 = normalize_scale(scale)

        mkpts0_c = data["mkpts0_c"] + delta_l * scale0
        mkpts1_c = data["mkpts1_c"] + delta_r * scale1

        if export_safe:
            data.update(
                {
                    "mkpts0_c": mkpts0_c,
                    "mkpts1_c": mkpts1_c,
                    "mkpts0_f": mkpts0_c,
                    "mkpts1_f": mkpts1_c,
                    "conf_matrix_f": softmax_matrix_f,
                }
            )
            return

        idx_r_iids = (idx_r // w).reshape(-1)
        idx_r_jids = (idx_r % w).reshape(-1)
        idx_l_flat = idx_l.reshape(-1)
        m_ids = torch.arange(m, device=idx_l.device, dtype=torch.long)
        delta = create_meshgrid(3, 3, True, conf_matrix_ff.device).to(torch.long)

        m_ids = m_ids[:, None, None].expand(-1, 3, 3)
        idx_l_map = idx_l_flat[:, None, None].expand(-1, 3, 3)
        idx_r_iids = idx_r_iids[:, None, None].expand(-1, 3, 3) + delta[None, ..., 1]
        idx_r_jids = idx_r_jids[:, None, None].expand(-1, 3, 3) + delta[None, ..., 0]
        idx_r_iids = torch.clamp(idx_r_iids, min=0, max=w + 1)
        idx_r_jids = torch.clamp(idx_r_jids, min=0, max=w + 1)

        conf_matrix_ff = conf_matrix_ff.reshape(m, self.WW, self.W + 2, self.W + 2)
        conf_matrix_ff = conf_matrix_ff[m_ids, idx_l_map, idx_r_iids, idx_r_jids]
        conf_matrix_ff = conf_matrix_ff.reshape(-1, 9)
        conf_matrix_ff = F.softmax(conf_matrix_ff / self.local_regress_temperature, -1)
        heatmap = conf_matrix_ff.reshape(-1, 3, 3)

        coords_normalized = dsnt.spatial_expectation2d(heatmap[None], True)[0]

        mkpts0_f = mkpts0_c
        mkpts1_f = mkpts1_c + (coords_normalized * (3 // 2) * scale1)

        data.update(
            {
                "mkpts0_c": mkpts0_c,
                "mkpts1_c": mkpts1_c,
                "mkpts0_f": mkpts0_f,
                "mkpts1_f": mkpts1_f,
                "conf_matrix_f": softmax_matrix_f,
            }
        )

    fine.forward = types.MethodType(export_friendly_forward, fine)


class BatchedTopKWrapper(nn.Module):
    """Wrap upstream EfficientLoFTR and emit fixed top-k outputs per batch item.

    Output tensors are compatible with simple batched consumers:
    - keypoints0: [B, K, 2]
    - keypoints1: [B, K, 2]
    - confidence: [B, K] (unused slots = -1)
    """

    def __init__(self, matcher: nn.Module, max_matches: int = 4096, max_batch: int = 16):
        super().__init__()
        self.matcher = matcher
        self.max_matches = int(max_matches)
        self.max_batch = int(max_batch)

    def forward(self, image0: torch.Tensor, image1: torch.Tensor):
        batch_size = image0.shape[0]
        if batch_size > self.max_batch:
            raise RuntimeError(
                f"batch size {batch_size} exceeds wrapper max_batch={self.max_batch}"
            )

        data = {"image0": image0, "image1": image1}
        self.matcher(data)

        mkpts0 = data["mkpts0_f"]
        mkpts1 = data["mkpts1_f"]
        mconf = data["mconf"]
        m_bids = data.get(
            "m_bids",
            torch.zeros((mconf.shape[0],), dtype=torch.long, device=mconf.device),
        )

        out0 = torch.zeros(
            (self.max_batch, self.max_matches, 2),
            dtype=mkpts0.dtype,
            device=mkpts0.device,
        )
        out1 = torch.zeros(
            (self.max_batch, self.max_matches, 2),
            dtype=mkpts1.dtype,
            device=mkpts1.device,
        )
        outc = torch.full(
            (self.max_batch, self.max_matches),
            -1.0,
            dtype=mconf.dtype,
            device=mconf.device,
        )

        bids = m_bids.to(torch.long)
        conf = mconf.to(torch.float32)
        # Stable ordering by (batch id asc, confidence desc).
        sort_key = bids.to(torch.float32) * 1_000_000.0 - conf
        order = torch.argsort(sort_key)

        mkpts0 = mkpts0[order]
        mkpts1 = mkpts1[order]
        mconf = mconf[order]
        bids = bids[order]

        one_hot = F.one_hot(bids, num_classes=self.max_batch).to(torch.long)
        rank_matrix = torch.cumsum(one_hot, dim=0) - 1
        row_ids = torch.arange(bids.shape[0], device=bids.device)
        ranks = rank_matrix[row_ids, bids]

        keep = (bids < batch_size) & (ranks < self.max_matches)
        flat = bids[keep] * self.max_matches + ranks[keep]

        out0.view(-1, 2)[flat] = mkpts0[keep]
        out1.view(-1, 2)[flat] = mkpts1[keep]
        outc.view(-1)[flat] = mconf[keep]

        return (
            out0[:batch_size],
            out1[:batch_size],
            outc[:batch_size],
        )