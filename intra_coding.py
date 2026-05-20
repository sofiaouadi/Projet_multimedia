"""
intra_coding.py — Part 2: Intra-frame Coding (I-Frames)
DCT-based spatial compression, analogous to JPEG.
Simplified MPEG-4 Video Encoder Pipeline
"""

import numpy as np
from scipy.fft import dct, idct
from tqdm import tqdm

from utils import QUANT_MATRIX_LUMA, pad_to_multiple, crop

# ─────────────────────────────────────────────────────────────────────────────
#  DCT / IDCT
# ─────────────────────────────────────────────────────────────────────────────

def dct2d(block: np.ndarray) -> np.ndarray:
    """
    2-D DCT-II (orthonormal) applied to an 8×8 block.
    Two successive 1-D DCT passes (rows then columns).
    """
    return dct(dct(block.T, norm='ortho').T, norm='ortho')


def idct2d(block: np.ndarray) -> np.ndarray:
    """Inverse 2-D DCT (IDCT-III, orthonormal)."""
    return idct(idct(block.T, norm='ortho').T, norm='ortho')

# ─────────────────────────────────────────────────────────────────────────────
#  QUANTISATION
# ─────────────────────────────────────────────────────────────────────────────

def get_quant_matrix(qf: float = 1.0) -> np.ndarray:
    """
    Return the quantisation matrix scaled by factor qf.
    qf > 1  → more aggressive quantisation (more compression, lower quality)
    qf < 1  → finer quantisation (less compression, higher quality)
    """
    return np.clip(QUANT_MATRIX_LUMA * qf, 1, None)   # avoid divide-by-zero


def quantise(dct_block: np.ndarray, qf: float = 1.0) -> np.ndarray:
    """Divide DCT coefficients by the quantisation matrix and round to int."""
    Q = get_quant_matrix(qf)
    return np.round(dct_block / Q).astype(np.int16)


def dequantise(q_block: np.ndarray, qf: float = 1.0) -> np.ndarray:
    """Multiply quantised coefficients back by the quantisation matrix."""
    Q = get_quant_matrix(qf)
    return (q_block.astype(np.float32) * Q)

# ─────────────────────────────────────────────────────────────────────────────
#  CHANNEL-LEVEL ENCODING / DECODING
# ─────────────────────────────────────────────────────────────────────────────

def encode_channel(channel: np.ndarray, qf: float = 1.0) -> tuple[np.ndarray, tuple]:
    """
    Split a single-channel image into 8×8 blocks, apply DCT + quantisation.

    Args:
        channel  — 2-D float32 array [0, 255]
        qf       — quantisation factor

    Returns:
        blocks     — int16 array of shape (n_rows, n_cols, 8, 8)
        orig_shape — (H, W) before padding, needed for exact reconstruction
    """
    orig_shape = channel.shape          # (H, W)
    ch = pad_to_multiple(channel, 8)   # pad to 8×8 multiple
    H, W = ch.shape
    n_rows, n_cols = H // 8, W // 8

    blocks = np.zeros((n_rows, n_cols, 8, 8), dtype=np.int16)

    for i in range(n_rows):
        for j in range(n_cols):
            raw   = ch[i*8:(i+1)*8, j*8:(j+1)*8].astype(np.float32) - 128.0
            dct_b = dct2d(raw)
            blocks[i, j] = quantise(dct_b, qf)

    return blocks, orig_shape


def decode_channel(blocks: np.ndarray, orig_shape: tuple,
                   qf: float = 1.0) -> np.ndarray:
    """
    Reconstruct a channel from quantised DCT blocks via dequantisation + IDCT.

    Args:
        blocks     — int16 array (n_rows, n_cols, 8, 8)
        orig_shape — (H, W) for cropping after reconstruction
        qf         — quantisation factor (must match encoder)

    Returns:
        channel  — float32 array clipped to [0, 255], shape orig_shape
    """
    n_rows, n_cols = blocks.shape[:2]
    recon = np.zeros((n_rows * 8, n_cols * 8), dtype=np.float32)

    for i in range(n_rows):
        for j in range(n_cols):
            dq           = dequantise(blocks[i, j], qf)
            pixel_block  = idct2d(dq) + 128.0
            recon[i*8:(i+1)*8, j*8:(j+1)*8] = pixel_block

    recon = np.clip(recon, 0, 255)
    return crop(recon, orig_shape)

# ─────────────────────────────────────────────────────────────────────────────
#  I-FRAME ENCODING / DECODING
# ─────────────────────────────────────────────────────────────────────────────

def encode_iframe(preprocessed: dict, qf: float = 1.0) -> dict:
    """
    Encode one I-frame (all three channels independently).

    Args:
        preprocessed — dict from preprocessing.preprocess_frame()
        qf           — quantisation factor

    Returns:
        Encoded frame dict with type='I' and compressed channel data.
    """
    Y_blocks,  Y_shape  = encode_channel(preprocessed['Y'],      qf)
    Cb_blocks, Cb_shape = encode_channel(preprocessed['Cb_sub'], qf)
    Cr_blocks, Cr_shape = encode_channel(preprocessed['Cr_sub'], qf)

    return {
        'type':  'I',
        'qf':    qf,
        'shape': preprocessed['shape'],   # original (H, W)
        'Y':  {'blocks': Y_blocks,  'orig_shape': Y_shape},
        'Cb': {'blocks': Cb_blocks, 'orig_shape': Cb_shape},
        'Cr': {'blocks': Cr_blocks, 'orig_shape': Cr_shape},
    }


def decode_iframe(data: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Decode an I-frame.

    Returns:
        Y    — float32 luma, full resolution
        Cb   — float32 blue-diff, subsampled (H/2 × W/2)
        Cr   — float32 red-diff,  subsampled (H/2 × W/2)
    """
    qf = data['qf']
    Y  = decode_channel(data['Y']['blocks'],  data['Y']['orig_shape'],  qf)
    Cb = decode_channel(data['Cb']['blocks'], data['Cb']['orig_shape'], qf)
    Cr = decode_channel(data['Cr']['blocks'], data['Cr']['orig_shape'], qf)
    return Y, Cb, Cr

# ─────────────────────────────────────────────────────────────────────────────
#  CONVENIENCE: ENCODE ALL I-FRAMES IN A SEQUENCE
# ─────────────────────────────────────────────────────────────────────────────

def encode_all_iframes(preprocessed_frames: list[dict],
                       qf: float = 1.0) -> list[dict]:
    """Encode every frame as an I-frame (used for testing Part 2 in isolation)."""
    encoded = []
    for pp in tqdm(preprocessed_frames, desc="[INTRA] Encoding I-frames", unit="fr"):
        encoded.append(encode_iframe(pp, qf))
    return encoded

# ─────────────────────────────────────────────────────────────────────────────
#  UTILITY: EXTRACT ONE BLOCK FOR VISUALISATION
# ─────────────────────────────────────────────────────────────────────────────

def get_block_stages(channel: np.ndarray,
                     block_row: int = 1,
                     block_col: int = 1,
                     qf: float = 1.0) -> dict:
    """
    Return the four stages of one 8×8 block for the visualisation figure:
        raw_pixels  → dct_coefficients → quantised → reconstructed

    Args:
        channel   — 2-D float32 luma channel
        block_row — which 8-px row of blocks to pick
        block_col — which 8-px column of blocks to pick
        qf        — quantisation factor
    """
    r, c = block_row * 8, block_col * 8
    raw   = channel[r:r+8, c:c+8].astype(np.float32)
    shifted = raw - 128.0
    dct_b = dct2d(shifted)
    q_b   = quantise(dct_b, qf)
    dq_b  = dequantise(q_b, qf)
    rec_b = np.clip(idct2d(dq_b) + 128.0, 0, 255)

    return {
        'raw':        raw,
        'dct':        dct_b,
        'quantised':  q_b.astype(np.float32),
        'reconstructed': rec_b,
    }
