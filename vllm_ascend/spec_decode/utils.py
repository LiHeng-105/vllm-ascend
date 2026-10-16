# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os

import torch

_fused_update_num_computed_tokens = None
_fused_import_attempted = False


def _get_fused_update():
    # Lazy import: a module-level import here creates a circular-import
    # hazard with vllm_ascend.device.device_op during package init.
    global _fused_update_num_computed_tokens, _fused_import_attempted
    if os.environ.get("VLLM_ASCEND_DISABLE_FUSED_NC_UPDATE"):
        return None
    if not _fused_import_attempted:
        _fused_import_attempted = True
        try:
            from vllm_ascend.ops.triton.update_num_computed import (
                update_num_computed_tokens as _fused_fn,
            )

            _fused_update_num_computed_tokens = _fused_fn
        except ImportError:
            _fused_update_num_computed_tokens = None
    return _fused_update_num_computed_tokens


def update_num_computed_tokens_for_batch_change(
    num_computed_tokens: torch.Tensor,
    num_accepted_tokens: torch.Tensor,
    prev_positions: torch.Tensor,
    valid_sampled_token_count: torch.Tensor,
    prev_num_draft_tokens: torch.Tensor,
    cpu_num_computed_tokens: torch.Tensor,
) -> None:
    """Correct num_computed_tokens for async spec decode drift.

    Requests that had drafts: corrected = prev_gpu + valid_count.
    New requests or non-draft (e.g. prefills): use CPU value directly.
    """
    n = prev_positions.shape[0]
    fused = _get_fused_update()
    if fused is not None and n <= 1024:
        fused(
            num_computed_tokens,
            num_accepted_tokens,
            prev_positions,
            valid_sampled_token_count,
            prev_num_draft_tokens,
            cpu_num_computed_tokens,
        )
        return
    # Clamp because prev_positions can be -1 for new requests
    gather_indices = prev_positions.clamp(min=0)

    valid_counts = valid_sampled_token_count[gather_indices]
    prev_computed = num_computed_tokens[gather_indices]
    prev_drafts = prev_num_draft_tokens[gather_indices]

    participating = (prev_positions >= 0) & (prev_drafts > 0)
    corrected = prev_computed + valid_counts.int()

    num_computed_tokens[:n].copy_(torch.where(participating, corrected, cpu_num_computed_tokens))
    num_accepted_tokens.copy_(torch.where(participating, valid_counts, num_accepted_tokens))
