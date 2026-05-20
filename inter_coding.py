"""
inter_coding.py — Part 3: Inter-frame Coding (P-Frames)
Motion estimation (full search MAD), residual DCT coding.
Simplified MPEG-4 Video Encoder Pipeline
"""

import numpy as np
from tqdm import tqdm

from intra_coding  import (encode_channel, decode_channel,
                            dct2d, idct2d, quantise, dequantise,
                            encode_iframe, decode_iframe)
from preprocessing import preprocess_frame
from utils         import pad_to_multiple, crop, ycbcr_to_bgr

# ─────────────────────────────────────────────────────────────────────────────
#  CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

MACROBLOCK_SIZE = 16   # pixels — standard MPEG macroblock


# ─────────────────────────────────────────────────────────────────────────────
#  MOTION ESTIMATION
# ─────────────────────────────────────────────────────────────────────────────

def full_search_block_match(cur_block: np.ndarray,
                             ref_frame: np.ndarray,
                             ref_y: int,
                             ref_x: int,
                             search: int) -> tuple[int, int]:
    """
    Full exhaustive search block matching using MAD (Mean Absolute Difference).

    For every candidate displacement (dy, dx) within ±search pixels of the
    anchor position (ref_y, ref_x), we compute the MAD between cur_block and
    the reference patch.  The displacement with the lowest MAD is returned as
    the motion vector.

    Args:
        cur_block  — current MB to match (MB×MB float32)
        ref_frame  — padded reference (reconstructed) luma frame
        ref_y/x    — anchor position in ref_frame
        search     — search window half-size in pixels

    Returns:
        (dy, dx) — best motion vector (signed integers, clamped to int8 range)
    """
    H, W = ref_frame.shape
    MB   = cur_block.shape[0]
    best_mad = np.inf
    best_dy, best_dx = 0, 0

    for dy in range(-search, search + 1):
        for dx in range(-search, search + 1):
            ry, rx = ref_y + dy, ref_x + dx
            # Skip out-of-bounds candidates
            if ry < 0 or rx < 0 or ry + MB > H or rx + MB > W:
                continue
            ref_block = ref_frame[ry:ry+MB, rx:rx+MB]
            mad = np.mean(np.abs(cur_block - ref_block))
            if mad < best_mad:
                best_mad  = mad
                best_dy   = dy
                best_dx   = dx

    # Clamp to int8 to save space (-128 … 127)
    best_dy = int(np.clip(best_dy, -128, 127))
    best_dx = int(np.clip(best_dx, -128, 127))
    return best_dy, best_dx


# ─────────────────────────────────────────────────────────────────────────────
#  RESIDUAL CODING HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _encode_residual_mb(residual: np.ndarray, qf: float) -> np.ndarray:
    """
    Encode one 16×16 residual macroblock by splitting into four 8×8 sub-blocks,
    applying DCT + quantisation to each.

    Returns int16 array of shape (4, 8, 8).
    """
    out = np.zeros((4, 8, 8), dtype=np.int16)
    k = 0
    for si in range(2):
        for sj in range(2):
            sub = residual[si*8:(si+1)*8, sj*8:(sj+1)*8]
            out[k] = quantise(dct2d(sub), qf)
            k += 1
    return out


def _decode_residual_mb(encoded_mb: np.ndarray, qf: float) -> np.ndarray:
    """
    Reconstruct one 16×16 residual macroblock from four encoded 8×8 sub-blocks.
    """
    recon = np.zeros((16, 16), dtype=np.float32)
    k = 0
    for si in range(2):
        for sj in range(2):
            dq = dequantise(encoded_mb[k], qf)
            recon[si*8:(si+1)*8, sj*8:(sj+1)*8] = idct2d(dq)
            k += 1
    return recon


# ─────────────────────────────────────────────────────────────────────────────
#  P-FRAME ENCODING / DECODING
# ─────────────────────────────────────────────────────────────────────────────

def encode_pframe(preprocessed: dict,
                  ref_Y_recon: np.ndarray,
                  qf: float = 1.0,
                  search: int = 8) -> dict:
    """
    Encode one P-frame.

    Steps:
        1. Divide Y channel into 16×16 macroblocks.
        2. For each MB: full-search block match against ref_Y_recon.
        3. Compute residual = current MB − motion-compensated prediction.
        4. Encode residual with DCT + quantisation (4 × 8×8 sub-blocks).
        5. Encode Cb / Cr channels intra (simpler, same as I-frame).

    Args:
        preprocessed — dict from preprocessing.preprocess_frame()
        ref_Y_recon  — reconstructed luma of the previous frame (float32)
        qf           — quantisation factor
        search       — motion search window (±search pixels)

    Returns:
        Encoded P-frame dict.
    """
    Y_cur = preprocessed['Y']
    shape = preprocessed['shape']   # (H, W)

    # Pad both current and reference to macroblock boundaries
    MB    = MACROBLOCK_SIZE
    Y_pad = pad_to_multiple(Y_cur,       MB).astype(np.float32)
    R_pad = pad_to_multiple(ref_Y_recon, MB).astype(np.float32)

    PH, PW = Y_pad.shape
    n_mby  = PH // MB
    n_mbx  = PW // MB

    motion_vectors   = np.zeros((n_mby, n_mbx, 2), dtype=np.int8)
    encoded_residuals = np.zeros((n_mby, n_mbx, 4, 8, 8), dtype=np.int16)

    for i in range(n_mby):
        for j in range(n_mbx):
            by, bx     = i * MB, j * MB
            cur_block  = Y_pad[by:by+MB, bx:bx+MB]

            dy, dx = full_search_block_match(cur_block, R_pad, by, bx, search)
            motion_vectors[i, j] = [dy, dx]

            pred = R_pad[by+dy:by+dy+MB, bx+dx:bx+dx+MB]
            residual = cur_block - pred

            encoded_residuals[i, j] = _encode_residual_mb(residual, qf)

    # Chroma: intra-coded (simpler, adequate for subsampled channels)
    Cb_blocks, Cb_shape = encode_channel(preprocessed['Cb_sub'], qf)
    Cr_blocks, Cr_shape = encode_channel(preprocessed['Cr_sub'], qf)

    return {
        'type':    'P',
        'qf':      qf,
        'shape':   shape,
        'mv':      motion_vectors,          # (n_mby, n_mbx, 2)  int8
        'res':     encoded_residuals,       # (n_mby, n_mbx, 4, 8, 8)  int16
        'Y_pad_shape': (PH, PW),
        'Cb': {'blocks': Cb_blocks, 'orig_shape': Cb_shape},
        'Cr': {'blocks': Cr_blocks, 'orig_shape': Cr_shape},
    }


def decode_pframe(data: dict,
                  ref_Y_recon: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Decode one P-frame.

    Steps:
        1. For each MB: fetch motion-compensated prediction from ref_Y_recon.
        2. Decode quantised residual (IDCT + dequantise).
        3. Reconstruct Y = prediction + residual.
        4. Decode Cb / Cr channels (intra).

    Returns:
        Y   — float32 reconstructed luma
        Cb  — float32 reconstructed blue-diff (subsampled)
        Cr  — float32 reconstructed red-diff  (subsampled)
    """
    qf  = data['qf']
    mv  = data['mv']
    res = data['res']
    MB  = MACROBLOCK_SIZE

    PH, PW = data['Y_pad_shape']
    R_pad  = pad_to_multiple(ref_Y_recon, MB).astype(np.float32)
    Y_recon_pad = np.zeros((PH, PW), dtype=np.float32)

    n_mby, n_mbx = mv.shape[:2]

    for i in range(n_mby):
        for j in range(n_mbx):
            by, bx = i * MB, j * MB
            dy = int(mv[i, j, 0])
            dx = int(mv[i, j, 1])

            pred     = R_pad[by+dy:by+dy+MB, bx+dx:bx+dx+MB]
            residual = _decode_residual_mb(res[i, j], qf)
            Y_recon_pad[by:by+MB, bx:bx+MB] = pred + residual

    Y = np.clip(crop(Y_recon_pad, data['shape']), 0, 255)

    Cb = decode_channel(data['Cb']['blocks'], data['Cb']['orig_shape'], qf)
    Cr = decode_channel(data['Cr']['blocks'], data['Cr']['orig_shape'], qf)

    return Y, Cb, Cr


# ─────────────────────────────────────────────────────────────────────────────
#  FULL GOP ENCODING
# ─────────────────────────────────────────────────────────────────────────────

def encode_gop(preprocessed_frames: list[dict],
               qf: float = 1.0,
               gop: int = 10,
               search: int = 8) -> tuple[list[dict], list[np.ndarray]]:
    """
    Encode all frames using a GOP structure.

    Every G-th frame (0, G, 2G, …) → I-frame.
    All others → P-frame referencing the previous reconstructed frame.

    Args:
        preprocessed_frames — list of dicts from preprocessing module
        qf                  — quantisation factor
        gop                 — Group of Pictures size
        search              — motion search window

    Returns:
        encoded_frames    — list of encoded frame dicts ('I' or 'P')
        ref_Y_recon_list  — list of reconstructed Y channels (for visualisation)
    """
    encoded_frames   = []
    ref_Y_recon_list = []
    prev_Y_recon     = None

    desc = f"[ENCODE] GOP={gop} QF={qf:.1f} Search=±{search}px"
    for idx, pp in enumerate(tqdm(preprocessed_frames, desc=desc, unit="fr")):

        is_iframe = (idx % gop == 0)

        if is_iframe:
            data = encode_iframe(pp, qf)
            Y_r, _, _ = decode_iframe(data)
        else:
            data = encode_pframe(pp, prev_Y_recon, qf, search)
            Y_r, _, _ = decode_pframe(data, prev_Y_recon)

        prev_Y_recon = Y_r.copy()
        encoded_frames.append(data)
        ref_Y_recon_list.append(Y_r)

    n_i = sum(1 for d in encoded_frames if d['type'] == 'I')
    n_p = len(encoded_frames) - n_i
    print(f"[ENCODE] Done — {n_i} I-frames  {n_p} P-frames")

    return encoded_frames, ref_Y_recon_list


# ─────────────────────────────────────────────────────────────────────────────
#  RESIDUAL MAP HELPER (for visualisation)
# ─────────────────────────────────────────────────────────────────────────────

def compute_residual_map(original_Y: np.ndarray,
                          recon_Y: np.ndarray) -> np.ndarray:
    """
    Return absolute residual map between original and reconstructed Y channels.
    Normalised to [0, 255] for display.
    """
    diff = np.abs(original_Y.astype(np.float32) - recon_Y.astype(np.float32))
    max_val = diff.max()
    if max_val > 0:
        diff = diff / max_val * 255.0
    return diff.astype(np.uint8)
