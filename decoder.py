"""
decoder.py — Full Decoding Pipeline
Reads a .bin file and reconstructs all video frames.
Simplified MPEG-4 Video Encoder Pipeline
"""

import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm

from entropy_coding import load_encoded
from intra_coding   import decode_iframe
from inter_coding   import decode_pframe
from utils          import ycbcr_to_bgr, save_frames


# ─────────────────────────────────────────────────────────────────────────────
#  FRAME RECONSTRUCTION
# ─────────────────────────────────────────────────────────────────────────────

def decode_frame(data: dict,
                 prev_Y_recon: np.ndarray | None) -> tuple[np.ndarray,
                                                            np.ndarray,
                                                            np.ndarray,
                                                            np.ndarray]:
    """
    Decode a single encoded frame (I or P type).

    Args:
        data         — encoded frame dict
        prev_Y_recon — reconstructed Y of the previous frame (None for I-frames)

    Returns:
        bgr_frame  — uint8 BGR image
        Y_recon    — float32 luma (used as reference for next P-frame)
        Cb_recon   — float32 subsampled blue-diff
        Cr_recon   — float32 subsampled red-diff
    """
    if data['type'] == 'I':
        Y, Cb, Cr = decode_iframe(data)

    elif data['type'] == 'P':
        if prev_Y_recon is None:
            raise RuntimeError("P-frame encountered before any I-frame — "
                               "stream is corrupted or truncated.")
        Y, Cb, Cr = decode_pframe(data, prev_Y_recon)

    else:
        raise ValueError(f"Unknown frame type: '{data['type']}'")

    # Convert YCbCr → BGR for display / saving
    bgr = ycbcr_to_bgr(Y, Cb, Cr)
    return bgr, Y, Cb, Cr


# ─────────────────────────────────────────────────────────────────────────────
#  FULL DECODE PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def decode_video(bin_path: Path,
                 output_folder: Path | None = None) -> list[np.ndarray]:
    """
    Full decoding pipeline.

    Args:
        bin_path      — path to the .bin file
        output_folder — if given, save reconstructed frames as PNG files here

    Returns:
        List of reconstructed BGR frames (uint8 numpy arrays).
    """
    bin_path = Path(bin_path)
    print(f"\n[DECODE] Reading '{bin_path}' …")
    encoded_frames = load_encoded(bin_path)
    n = len(encoded_frames)
    print(f"[DECODE] {n} frames to decode")

    recon_bgr_frames  = []
    prev_Y_recon      = None

    for idx, data in enumerate(tqdm(encoded_frames,
                                     desc="[DECODE] Reconstructing frames",
                                     unit="fr")):
        bgr, Y, Cb, Cr = decode_frame(data, prev_Y_recon)
        prev_Y_recon   = Y.copy()
        recon_bgr_frames.append(bgr)

    n_i = sum(1 for d in encoded_frames if d['type'] == 'I')
    n_p = n - n_i
    print(f"[DECODE] Done — {n_i} I-frames  {n_p} P-frames reconstructed")

    if output_folder is not None:
        save_frames(recon_bgr_frames, Path(output_folder), prefix="recon")

    return recon_bgr_frames


# ─────────────────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="MPEG-4 Simplified Decoder")
    parser.add_argument('bin',    help="Input .bin file")
    parser.add_argument('outdir', help="Output folder for reconstructed frames")
    args = parser.parse_args()

    decode_video(Path(args.bin), Path(args.outdir))
