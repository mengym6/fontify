"""Pixel-mask geometry shared by Run B corruption, loss and previews."""

def is_pixel_mask(mask, target):
    """Accept either BxHxW or the trainer's flattened Bx(HW) mask."""
    return mask.shape[0] == target.shape[0] and mask[0].numel() == (
        target.shape[-2] * target.shape[-1]
    )


def expand_mask(mask, target, patch_size):
    """Return Bx3xHxW without expanding semantic pixels to whole patches."""
    batch, channels, height, width = target.shape
    if is_pixel_mask(mask, target):
        pixels = mask.reshape(batch, height, width)
    else:
        pixels = mask.reshape(batch, height // patch_size, width // patch_size)
        pixels = pixels.repeat_interleave(patch_size, 1).repeat_interleave(
            patch_size, 2
        )
    return pixels[:, None].expand(-1, channels, -1, -1)


def corrupt_before_embedding(target, mask, patch_size):
    """Remove hidden GT first; use mask tokens only for entirely hidden patches."""
    pixels = expand_mask(mask, target, patch_size)[:, 0].bool()
    batch, height, width = pixels.shape
    fully_hidden = pixels.reshape(
        batch, height // patch_size, patch_size, width // patch_size, patch_size
    ).all(4).all(2)
    return target.masked_fill(pixels[:, None], 0), fully_hidden.flatten(1)
