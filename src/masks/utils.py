# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch


def apply_masks(x, masks, concat=True):
    """
    :param x: tensor of shape [B (batch-size), N (num-patches), D (feature-dim)]
    :param masks: list of tensors of shape [B, K] containing indices of K patches in [N] to keep
    """
    all_x = []
    for m in masks:
        mask_keep = m.unsqueeze(-1)
        if m.ndim == 1:
            mask_keep = mask_keep.unsqueeze(0)
        mask_keep = mask_keep.expand(*mask_keep.shape[:-1], x.size(-1))
        all_x += [torch.gather(x, dim=1, index=mask_keep)]
    if not concat:
        return all_x
    if len(all_x) == 1:
        return all_x[0]

    return torch.cat(all_x, dim=0)
