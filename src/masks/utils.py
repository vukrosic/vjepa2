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
    if not masks:
        return [] if not concat else x.new_empty((0, 0, x.size(-1)))

    if len(masks) == 1:
        m = masks[0]
        mask_keep = m.unsqueeze(-1)
        if m.ndim == 1:
            mask_keep = mask_keep.unsqueeze(0)
        mask_keep = mask_keep.expand(*mask_keep.shape[:-1], x.size(-1))
        return torch.gather(x, dim=1, index=mask_keep) if concat else [torch.gather(x, dim=1, index=mask_keep)]

    same_shape_2d = masks[0].ndim == 2 and all(m.ndim == 2 and m.shape == masks[0].shape for m in masks)
    if same_shape_2d:
        stacked_masks = torch.stack(masks, dim=0)
        gathered = torch.gather(
            x.unsqueeze(0).expand(len(masks), -1, -1, -1),
            dim=2,
            index=stacked_masks.unsqueeze(-1).expand(-1, -1, -1, x.size(-1)),
        )
        if not concat:
            return list(gathered.unbind(0))
        return gathered.reshape(-1, stacked_masks.size(-1), x.size(-1))

    all_x = []
    for m in masks:
        mask_keep = m.unsqueeze(-1)
        if m.ndim == 1:
            mask_keep = mask_keep.unsqueeze(0)
        mask_keep = mask_keep.expand(*mask_keep.shape[:-1], x.size(-1))
        all_x += [torch.gather(x, dim=1, index=mask_keep)]
    if not concat:
        return all_x

    return torch.cat(all_x, dim=0)
