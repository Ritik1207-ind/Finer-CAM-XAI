import torch


def reshape_transform(tensor: torch.Tensor, H=None, W=None) -> torch.Tensor:
    """
    Convert ViT token activations from:
        [B, 197, 768]

    to:
        [B, 768, 14, 14]

    The first token is the CLS token, so it is removed.
    The remaining 196 tokens correspond to a 14x14 patch grid.
    """
    if tensor.ndim != 3:
        raise ValueError(
            f"Expected 3D tensor [B, tokens, channels], got {tensor.shape}"
        )

    batch_size, num_tokens, channels = tensor.shape

    if num_tokens != 197:
        raise ValueError(
            f"Expected 197 tokens (1 CLS + 196 patches), got {num_tokens}"
        )

    patch_tokens = tensor[:, 1:, :]  # remove CLS token

    patch_tokens = patch_tokens.reshape(
        batch_size, 14, 14, channels
    )

    # [B, 14, 14, 768] -> [B, 768, 14, 14]
    patch_tokens = patch_tokens.permute(0, 3, 1, 2)

    return patch_tokens
