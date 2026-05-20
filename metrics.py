"""
metrics.py — Part 5a: Quality & Compression Metrics
PSNR, SSIM, MSE, compression ratio, I/P frame counts.
Simplified MPEG-4 Video Encoder Pipeline
"""

import numpy as np
from pathlib import Path


# ─────────────────────────────────────────────────────────────────────────────
#  PER-FRAME METRICS
# ─────────────────────────────────────────────────────────────────────────────

def compute_mse(orig: np.ndarray, recon: np.ndarray) -> float:
    """Mean Squared Error between two uint8/float arrays."""
    return float(np.mean(
        (orig.astype(np.float64) - recon.astype(np.float64)) ** 2
    ))


def compute_psnr(orig: np.ndarray, recon: np.ndarray,
                 max_val: float = 255.0) -> float:
    """
    Peak Signal-to-Noise Ratio (dB).
    PSNR = 10 · log₁₀(MAX² / MSE)
    Higher is better. > 30 dB is generally acceptable.
    Returns inf if MSE == 0 (perfect reconstruction).
    """
    mse = compute_mse(orig, recon)
    if mse == 0:
        return float('inf')
    return 10.0 * np.log10(max_val ** 2 / mse)


def compute_ssim(orig: np.ndarray, recon: np.ndarray) -> float:
    """
    Simplified global SSIM (Structural Similarity Index).
    Uses the global mean/variance — not the windowed version —
    which is sufficient for project-level quality assessment.

    Returns a value in [-1, 1]; closer to 1 = better quality.
    """
    C1 = (0.01 * 255) ** 2   # stability constants (Wang et al. 2004)
    C2 = (0.03 * 255) ** 2

    o = orig.astype(np.float64)
    r = recon.astype(np.float64)

    mu_o    = o.mean()
    mu_r    = r.mean()
    sigma_o = o.var()
    sigma_r = r.var()
    sigma_or = np.mean((o - mu_o) * (r - mu_r))

    num = (2 * mu_o * mu_r + C1) * (2 * sigma_or + C2)
    den = (mu_o**2 + mu_r**2 + C1) * (sigma_o + sigma_r + C2)
    return float(num / den)

# ─────────────────────────────────────────────────────────────────────────────
#  COMPRESSION METRICS
# ─────────────────────────────────────────────────────────────────────────────

def original_size_bytes(frames: list[np.ndarray]) -> int:
    """Total raw size of all frames in bytes (H × W × C × dtype_bytes)."""
    return sum(f.nbytes for f in frames)


def compressed_size_bytes(bin_path: Path) -> int:
    """Size of the .bin output file in bytes."""
    return Path(bin_path).stat().st_size


def compression_ratio(original_bytes: int, compressed_bytes: int) -> float:
    """
    Compression ratio = original / compressed.
    ratio > 1 means compression is effective.
    """
    return original_bytes / max(compressed_bytes, 1)

# ─────────────────────────────────────────────────────────────────────────────
#  FULL METRICS REPORT
# ─────────────────────────────────────────────────────────────────────────────

def compute_all_metrics(original_frames: list[np.ndarray],
                         recon_frames: list[np.ndarray],
                         encoded_frames: list[dict],
                         bin_path: Path) -> dict:
    """
    Compute the complete set of metrics for an encoded/decoded video.

    Returns a dict with per-frame lists and global averages.
    """
    assert len(original_frames) == len(recon_frames), "Frame count mismatch"
    n = len(original_frames)

    psnr_list  = []
    mse_list   = []
    ssim_list  = []
    types      = []

    for orig, recon, enc in zip(original_frames, recon_frames, encoded_frames):
        psnr_list.append(compute_psnr(orig, recon))
        mse_list.append(compute_mse(orig, recon))
        ssim_list.append(compute_ssim(orig, recon))
        types.append(enc['type'])

    orig_bytes = original_size_bytes(original_frames)
    comp_bytes = compressed_size_bytes(bin_path)
    ratio      = compression_ratio(orig_bytes, comp_bytes)

    n_i = types.count('I')
    n_p = types.count('P')

    metrics = {
        'n_frames':          n,
        'n_iframes':         n_i,
        'n_pframes':         n_p,
        'frame_types':       types,
        'psnr':              psnr_list,
        'mse':               mse_list,
        'ssim':              ssim_list,
        'avg_psnr':          float(np.mean(psnr_list)),
        'avg_mse':           float(np.mean(mse_list)),
        'avg_ssim':          float(np.mean(ssim_list)),
        'original_bytes':    orig_bytes,
        'compressed_bytes':  comp_bytes,
        'compression_ratio': ratio,
    }

    # ── Pretty print ──────────────────────────────────────────────
    bar = "═" * 50
    print(f"\n{bar}")
    print(f"  METRICS SUMMARY")
    print(bar)
    print(f"  Frames          : {n}  ({n_i} I-frames + {n_p} P-frames)")
    print(f"  Original size   : {orig_bytes / 1024:.1f} KB")
    print(f"  Compressed size : {comp_bytes / 1024:.1f} KB")
    print(f"  Compression ratio: {ratio:.2f}×")
    print(f"  Avg PSNR        : {metrics['avg_psnr']:.2f} dB")
    print(f"  Avg SSIM        : {metrics['avg_ssim']:.4f}")
    print(f"  Avg MSE         : {metrics['avg_mse']:.2f}")
    print(bar + "\n")

    return metrics


# ─────────────────────────────────────────────────────────────────────────────
#  EXPERIMENTAL SWEEPS
# ─────────────────────────────────────────────────────────────────────────────

def sweep_qf(frames_bgr: list[np.ndarray],
             qf_values: list[float],
             gop: int = 10,
             search: int = 4,
             tmp_bin: Path = Path('/tmp/_sweep_qf.bin')) -> dict:
    """
    Run the full encode → decode pipeline for each QF value.
    Returns a dict mapping qf → {'ratio', 'avg_psnr', 'avg_ssim'}.
    """
    from preprocessing import preprocess_all
    from inter_coding  import encode_gop
    from intra_coding  import decode_iframe
    from inter_coding  import decode_pframe
    from entropy_coding import save_encoded, load_encoded
    from decoder        import decode_video

    results = {}
    pp_frames = preprocess_all(frames_bgr)

    for qf in qf_values:
        print(f"\n── QF sweep: qf={qf:.1f} ──")
        enc_frames, _ = encode_gop(pp_frames, qf=qf, gop=gop, search=search)
        save_encoded(enc_frames, tmp_bin)
        recon = decode_video(tmp_bin)

        orig_b = original_size_bytes(frames_bgr)
        comp_b = compressed_size_bytes(tmp_bin)
        ratio  = compression_ratio(orig_b, comp_b)
        avg_psnr = float(np.mean([compute_psnr(o, r)
                                   for o, r in zip(frames_bgr, recon)]))
        avg_ssim = float(np.mean([compute_ssim(o, r)
                                   for o, r in zip(frames_bgr, recon)]))

        results[qf] = {'ratio': ratio, 'avg_psnr': avg_psnr, 'avg_ssim': avg_ssim}
        print(f"   ratio={ratio:.2f}×  PSNR={avg_psnr:.2f} dB  SSIM={avg_ssim:.4f}")

    return results


def sweep_gop(frames_bgr: list[np.ndarray],
              gop_values: list[int],
              qf: float = 1.0,
              search: int = 4,
              tmp_bin: Path = Path('/tmp/_sweep_gop.bin')) -> dict:
    """
    Run the full encode → decode pipeline for each GOP size.
    Returns a dict mapping gop → compression_ratio.
    """
    from preprocessing  import preprocess_all
    from inter_coding   import encode_gop
    from entropy_coding import save_encoded

    results   = {}
    pp_frames = preprocess_all(frames_bgr)

    for g in gop_values:
        print(f"\n── GOP sweep: gop={g} ──")
        enc_frames, _ = encode_gop(pp_frames, qf=qf, gop=g, search=search)
        save_encoded(enc_frames, tmp_bin)
        ratio = compression_ratio(original_size_bytes(frames_bgr),
                                   compressed_size_bytes(tmp_bin))
        results[g] = ratio
        print(f"   GOP={g}  ratio={ratio:.2f}×")

    return results
