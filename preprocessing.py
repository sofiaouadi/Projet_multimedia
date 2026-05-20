"""
preprocessing.py — Part 1: Color Space Conversion & Chroma Subsampling
Simplified MPEG-4 Video Encoder Pipeline
"""

import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm


# ─────────────────────────────────────────────────────────────────────────────
#  COLOR SPACE CONVERSION
# ─────────────────────────────────────────────────────────────────────────────

def bgr_to_ycbcr(frame_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert a BGR uint8 frame to YCbCr float32 channels.

    Formulas (ITU-R BT.601):
        Y  =  0.299·R + 0.587·G + 0.114·B
        Cb = −0.169·R − 0.331·G + 0.500·B + 128
        Cr =  0.500·R − 0.419·G − 0.081·B + 128

    Returns:
        Y  — luminance channel  [0, 255]
        Cb — blue-difference    [0, 255]
        Cr — red-difference     [0, 255]
    """
    frame_bgr = frame_bgr.astype(np.float32)
    B = frame_bgr[:, :, 0]
    G = frame_bgr[:, :, 1]
    R = frame_bgr[:, :, 2]

    Y  =  0.299  * R + 0.587  * G + 0.114  * B
    Cb = -0.16875 * R - 0.33126 * G + 0.5   * B + 128.0
    Cr =  0.5    * R - 0.41869 * G - 0.08131 * B + 128.0

    return Y, Cb, Cr


def ycbcr_to_bgr_full(Y: np.ndarray,
                       Cb: np.ndarray,
                       Cr: np.ndarray) -> np.ndarray:
    """
    Convert full-resolution YCbCr float32 channels back to BGR uint8.
    (Used when Cb/Cr have NOT been subsampled yet.)
    """
    Cb_s = Cb - 128.0
    Cr_s = Cr - 128.0

    R = np.clip(Y + 1.402   * Cr_s,              0, 255)
    G = np.clip(Y - 0.34414 * Cb_s - 0.71414 * Cr_s, 0, 255)
    B = np.clip(Y + 1.772   * Cb_s,              0, 255)

    return np.stack([B, G, R], axis=-1).astype(np.uint8)


# ─────────────────────────────────────────────────────────────────────────────
#  4:2:0 CHROMA SUBSAMPLING
# ─────────────────────────────────────────────────────────────────────────────

def chroma_subsample_420(Cb: np.ndarray,
                          Cr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    4:2:0 chroma subsampling.
    Halve both spatial dimensions of Cb and Cr by averaging 2×2 blocks.
    This discards 75 % of chrominance samples with minimal visual loss
    because the HVS is far more sensitive to luma than chroma.
    """
    # Box-filter (average) before downsampling — avoids aliasing
    Cb_sub = (Cb[0::2, 0::2] + Cb[1::2, 0::2] +
              Cb[0::2, 1::2] + Cb[1::2, 1::2]) / 4.0
    Cr_sub = (Cr[0::2, 0::2] + Cr[1::2, 0::2] +
              Cr[0::2, 1::2] + Cr[1::2, 1::2]) / 4.0
    return Cb_sub, Cr_sub


def chroma_upsample_420(Cb_sub: np.ndarray,
                         Cr_sub: np.ndarray,
                         target_shape: tuple) -> tuple[np.ndarray, np.ndarray]:
    """
    4:2:0 chroma upsampling (nearest-neighbour ×2 in each dimension).
    Crops to target_shape (H, W) to handle odd dimensions.
    """
    h, w = target_shape
    Cb_up = np.repeat(np.repeat(Cb_sub, 2, axis=0), 2, axis=1)[:h, :w]
    Cr_up = np.repeat(np.repeat(Cr_sub, 2, axis=0), 2, axis=1)[:h, :w]
    return Cb_up, Cr_up


# ─────────────────────────────────────────────────────────────────────────────
#  FULL PREPROCESSING PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def preprocess_frame(frame_bgr: np.ndarray) -> dict:
    """
    Apply full pre-processing to one BGR frame:
        1. BGR → YCbCr
        2. 4:2:0 chroma subsampling on Cb and Cr

    Returns a dict with:
        'Y'        — full-res luma         float32
        'Cb'       — full-res blue-diff    float32
        'Cr'       — full-res red-diff     float32
        'Cb_sub'   — subsampled Cb (H/2 × W/2) float32
        'Cr_sub'   — subsampled Cr (H/2 × W/2) float32
        'shape'    — (H, W) original frame shape
    """
    Y, Cb, Cr = bgr_to_ycbcr(frame_bgr)
    Cb_sub, Cr_sub = chroma_subsample_420(Cb, Cr)

    return {
        'Y':      Y,
        'Cb':     Cb,
        'Cr':     Cr,
        'Cb_sub': Cb_sub,
        'Cr_sub': Cr_sub,
        'shape':  frame_bgr.shape[:2],   # (H, W)
    }


def preprocess_all(frames: list[np.ndarray]) -> list[dict]:
    """Pre-process every frame in the list (with progress bar)."""
    result = []
    for frame in tqdm(frames, desc="[PRE-PROCESS] Converting BGR → YCbCr + 4:2:0", unit="fr"):
        result.append(preprocess_frame(frame))
    return result
