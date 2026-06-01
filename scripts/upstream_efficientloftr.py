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
    _patch_rope_rotate_half_for_export(matcher)
    _patch_loftr_forward_for_export(matcher)
    _patch_coarse_matching_for_export(matcher)
    _patch_fine_preprocess_for_export(matcher)
    _patch_fine_matching_for_export(matcher, export_safe=export_safe)
    matcher.eval()
    return matcher


def _patch_rope_rotate_half_for_export(matcher: nn.Module) -> None:
    def rotate_half_no_split(self, x):
        x1 = x[..., 0::2]
        x2 = x[..., 1::2]
        out = torch.empty_like(x)
        out[..., 0::2] = -x2
        out[..., 1::2] = x1
        return out

    for layer in cast(Any, matcher).loftr_coarse.layers:
        if hasattr(layer, "rope_pos_enc"):
            layer.rope_pos_enc.rotate_half = types.MethodType(rotate_half_no_split, layer.rope_pos_enc)


def _patch_loftr_forward_for_export(matcher: nn.Module) -> None:
    def export_friendly_forward(self, data):
        data.update(
            {
                "bs": data["image0"].size(0),
                "hw0_i": data["image0"].shape[2:],
                "hw1_i": data["image1"].shape[2:],
            }
        )

        if data["hw0_i"] == data["hw1_i"]:
            ret_dict = self.backbone(torch.cat([data["image0"], data["image1"]], dim=0))
            feats_c = ret_dict["feats_c"]
            data.update({"feats_x2": ret_dict["feats_x2"], "feats_x1": ret_dict["feats_x1"]})
            bs = data["bs"]
            feat_c0 = feats_c[:bs]
            feat_c1 = feats_c[bs:]
        else:
            ret_dict0, ret_dict1 = self.backbone(data["image0"]), self.backbone(data["image1"])
            feat_c0 = ret_dict0["feats_c"]
            feat_c1 = ret_dict1["feats_c"]
            data.update(
                {
                    "feats_x2_0": ret_dict0["feats_x2"],
                    "feats_x1_0": ret_dict0["feats_x1"],
                    "feats_x2_1": ret_dict1["feats_x2"],
                    "feats_x1_1": ret_dict1["feats_x1"],
                }
            )

        mul = self.config["resolution"][0] // self.config["resolution"][1]
        data.update(
            {
                "hw0_c": feat_c0.shape[2:],
                "hw1_c": feat_c1.shape[2:],
                "hw0_f": [feat_c0.shape[2] * mul, feat_c0.shape[3] * mul],
                "hw1_f": [feat_c1.shape[2] * mul, feat_c1.shape[3] * mul],
            }
        )

        mask_c0 = mask_c1 = None
        if "mask0" in data:
            mask_c0, mask_c1 = data["mask0"], data["mask1"]

        feat_c0, feat_c1 = self.loftr_coarse(feat_c0, feat_c1, mask_c0, mask_c1)
        feat_c0 = rearrange(feat_c0, "n c h w -> n (h w) c")
        feat_c1 = rearrange(feat_c1, "n c h w -> n (h w) c")

        self.coarse_matching(
            feat_c0,
            feat_c1,
            data,
            mask_c0=mask_c0.view(mask_c0.size(0), -1) if mask_c0 is not None else mask_c0,
            mask_c1=mask_c1.view(mask_c1.size(0), -1) if mask_c1 is not None else mask_c1,
        )

        feat_c0, feat_c1 = map(lambda feat: feat / feat.shape[-1] ** 0.5, [feat_c0, feat_c1])
        feat_f0_unfold, feat_f1_unfold = self.fine_preprocess(feat_c0, feat_c1, data)
        del feat_c0, feat_c1, mask_c0, mask_c1
        self.fine_matching(feat_f0_unfold, feat_f1_unfold, data)

    matcher.forward = types.MethodType(export_friendly_forward, matcher)


def _patch_coarse_matching_for_export(matcher: nn.Module) -> None:
    from src.loftr.utils.coarse_matching import (
        compute_max_candidates,
        mask_border,
        mask_border_with_padding,
    )

    coarse = cast(Any, matcher).coarse_matching

    def export_friendly_get_coarse_match(self, conf_matrix, data):
        axes_lengths = {
            "h0c": data["hw0_c"][0],
            "w0c": data["hw0_c"][1],
            "h1c": data["hw1_c"][0],
            "w1c": data["hw1_c"][1],
        }
        _device = conf_matrix.device
        mask = conf_matrix > self.thr
        mask = rearrange(
            mask,
            "b (h0c w0c) (h1c w1c) -> b h0c w0c h1c w1c",
            **axes_lengths,
        )

        if "mask0" not in data:
            mask_border(mask, self.border_rm, False)
        else:
            mask_border_with_padding(mask, self.border_rm, False, data["mask0"], data["mask1"])
        mask = rearrange(
            mask,
            "b h0c w0c h1c w1c -> b (h0c w0c) (h1c w1c)",
            **axes_lengths,
        )

        mask = mask * (conf_matrix == conf_matrix.max(dim=2, keepdim=True)[0]) * (
            conf_matrix == conf_matrix.max(dim=1, keepdim=True)[0]
        )

        mask_int = mask.to(torch.int64)
        mask_v, all_j_ids = mask_int.max(dim=2)
        b_ids, i_ids = torch.where(mask_v > 0)
        j_ids = all_j_ids[b_ids, i_ids]
        mconf = conf_matrix[b_ids, i_ids, j_ids]

        if self.training:
            if "mask0" not in data:
                num_candidates_max = mask.size(0) * max(mask.size(1), mask.size(2))
            else:
                num_candidates_max = compute_max_candidates(data["mask0"], data["mask1"])
            num_matches_train = int(num_candidates_max * self.train_coarse_percent)
            num_matches_pred = len(b_ids)
            assert self.train_pad_num_gt_min < num_matches_train, "min-num-gt-pad should be less than num-train-matches"

            if num_matches_pred <= num_matches_train - self.train_pad_num_gt_min:
                pred_indices = torch.arange(num_matches_pred, device=_device)
            else:
                pred_indices = torch.randint(
                    num_matches_pred,
                    (num_matches_train - self.train_pad_num_gt_min,),
                    device=_device,
                )

            gt_pad_indices = torch.randint(
                len(data["spv_b_ids"]),
                (max(num_matches_train - num_matches_pred, self.train_pad_num_gt_min),),
                device=_device,
            )
            mconf_gt = torch.zeros(len(data["spv_b_ids"]), device=_device)

            b_ids, i_ids, j_ids, mconf = map(
                lambda x, y: torch.cat([x[pred_indices], y[gt_pad_indices]], dim=0),
                *zip(
                    [b_ids, data["spv_b_ids"]],
                    [i_ids, data["spv_i_ids"]],
                    [j_ids, data["spv_j_ids"]],
                    [mconf, mconf_gt],
                ),
            )

        coarse_matches = {"b_ids": b_ids, "i_ids": i_ids, "j_ids": j_ids}

        scale = data["hw0_i"][0] / data["hw0_c"][0]

        scale0 = scale * data["scale0"][b_ids] if "scale0" in data else scale
        scale1 = scale * data["scale1"][b_ids] if "scale1" in data else scale
        mkpts0_c = torch.stack(
            [i_ids % data["hw0_c"][1], i_ids // data["hw0_c"][1]],
            dim=1,
        ) * scale0
        mkpts1_c = torch.stack(
            [j_ids % data["hw1_c"][1], j_ids // data["hw1_c"][1]],
            dim=1,
        ) * scale1

        m_bids = b_ids[mconf != 0]
        coarse_matches.update(
            {
                "m_bids": m_bids,
                "mkpts0_c": mkpts0_c[mconf != 0],
                "mkpts1_c": mkpts1_c[mconf != 0],
                "mconf": mconf[mconf != 0],
            }
        )

        return coarse_matches

    coarse.get_coarse_match = types.MethodType(export_friendly_get_coarse_match, coarse)


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
            bs = data["bs"]
            feat_f0 = x1[:bs]
            feat_f1 = x1[bs:]

            feat_f0 = F.unfold(feat_f0, kernel_size=(w, w), stride=stride, padding=0)
            feat_f0 = rearrange(feat_f0, "n (c ww) l -> n l ww c", ww=w**2)
            feat_f1 = F.unfold(feat_f1, kernel_size=(w + 2, w + 2), stride=stride, padding=1)
            feat_f1 = rearrange(feat_f1, "n (c ww) l -> n l ww c", ww=(w + 2) ** 2)

            feat_f0 = feat_f0.reshape(-1, feat_f0.shape[2], feat_f0.shape[3])
            feat_f1 = feat_f1.reshape(-1, feat_f1.shape[2], feat_f1.shape[3])
            feat_f0 = torch.index_select(
                feat_f0,
                0,
                (data["b_ids"] * data["hw0_c"][1] + data["i_ids"]).to(torch.long),
            )
            feat_f1 = torch.index_select(
                feat_f1,
                0,
                (data["b_ids"] * data["hw1_c"][1] + data["j_ids"]).to(torch.long),
            )
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

        feat_f0 = feat_f0.reshape(-1, feat_f0.shape[2], feat_f0.shape[3])
        feat_f1 = feat_f1.reshape(-1, feat_f1.shape[2], feat_f1.shape[3])
        feat_f0 = torch.index_select(
            feat_f0,
            0,
            (data["b_ids"] * data["hw0_c"][1] + data["i_ids"]).to(torch.long),
        )
        feat_f1 = torch.index_select(
            feat_f1,
            0,
            (data["b_ids"] * data["hw1_c"][1] + data["j_ids"]).to(torch.long),
        )
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

        conf_flat = softmax_matrix_f.reshape(m, ww * ww)
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
        data = {"image0": image0, "image1": image1}
        self.matcher(data)

        mkpts0 = data["mkpts0_f"]
        mkpts1 = data["mkpts1_f"]
        mconf = data["mconf"]
        m_bids = data.get(
            "m_bids",
            torch.zeros((mconf.shape[0],), dtype=torch.long, device=mconf.device),
        )

        bids = m_bids.to(torch.long)
        conf = mconf.to(torch.float32)
        neg_inf = torch.tensor(-1e9, dtype=conf.dtype, device=conf.device)
        out0_rows = []
        out1_rows = []
        outc_rows = []

        for batch_idx in range(self.max_batch):
            batch_mask = bids == batch_idx
            batch_scores = torch.where(batch_mask, conf, neg_inf)
            k = min(self.max_matches, int(batch_scores.numel()))
            topk_scores, topk_indices = torch.topk(batch_scores, k=k)
            if k < self.max_matches:
                pad = self.max_matches - k
                topk_scores = torch.cat(
                    [topk_scores, torch.full((pad,), neg_inf, dtype=topk_scores.dtype, device=topk_scores.device)]
                )
                topk_indices = torch.cat(
                    [topk_indices, torch.zeros((pad,), dtype=topk_indices.dtype, device=topk_indices.device)]
                )
            valid = topk_scores > (neg_inf * 0.5)
            selected0 = mkpts0[topk_indices]
            selected1 = mkpts1[topk_indices]
            selectedc = topk_scores
            out0_rows.append(torch.where(valid.unsqueeze(-1), selected0, torch.zeros_like(selected0)))
            out1_rows.append(torch.where(valid.unsqueeze(-1), selected1, torch.zeros_like(selected1)))
            outc_rows.append(torch.where(valid, selectedc, torch.full_like(selectedc, -1.0)))

        return torch.stack(out0_rows, dim=0), torch.stack(out1_rows, dim=0), torch.stack(outc_rows, dim=0)