"""
utils.py — Shared utilities, constants, and helpers
Simplified MPEG-4 Video Encoder Pipeline
"""

from pathlib import Path
import numpy as np
import cv2

# ─────────────────────────────────────────────────────────────────────────────
#  PATHS
# ─────────────────────────────────────────────────────────────────────────────

PROJECT_ROOT  = Path(__file__).resolve().parent.parent
DATA_DIR      = PROJECT_ROOT / "data"
INPUT_DIR     = DATA_DIR / "input_frames"
OUTPUT_DIR    = DATA_DIR / "output"
REPORT_DIR    = PROJECT_ROOT / "report"

def ensure_dirs():
    """Create all required directories if they don't exist."""
    for d in [INPUT_DIR, OUTPUT_DIR, REPORT_DIR]:
        d.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
#  STANDARD JPEG LUMA QUANTISATION MATRIX
# ─────────────────────────────────────────────────────────────────────────────

QUANT_MATRIX_LUMA = np.array([
    [16, 11, 10, 16,  24,  40,  51,  61],
    [12, 12, 14, 19,  26,  58,  60,  55],
    [14, 13, 16, 24,  40,  57,  69,  56],
    [14, 17, 22, 29,  51,  87,  80,  62],
    [18, 22, 37, 56,  68, 109, 103,  77],
    [24, 35, 55, 64,  81, 104, 113,  92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103,  99],
], dtype=np.float32)

# ─────────────────────────────────────────────────────────────────────────────
#  FRAME I/O
# ─────────────────────────────────────────────────────────────────────────────

def load_frames(folder: Path) -> list[np.ndarray]:
    """
    Load all PNG/JPG frames from a folder, sorted by filename.
    Returns a list of BGR uint8 arrays.
    """
    folder = Path(folder)
    exts   = {'.png', '.jpg', '.jpeg'}
    files  = sorted(p for p in folder.iterdir() if p.suffix.lower() in exts)

    if not files:
        raise FileNotFoundError(f"No image frames found in '{folder}'")

    frames = []
    for f in files:
        img = cv2.imread(str(f))
        if img is None:
            raise IOError(f"Cannot read '{f}'")
        frames.append(img)

    h, w = frames[0].shape[:2]
    print(f"[UTILS] Loaded {len(frames)} frames  ({w}×{h} px)")
    return frames


def save_frames(frames: list[np.ndarray], folder: Path, prefix: str = "frame"):
    """Save a list of BGR frames as PNG files."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    for i, f in enumerate(frames):
        cv2.imwrite(str(folder / f"{prefix}_{i:04d}.png"), f)
    print(f"[UTILS] Saved {len(frames)} frames → '{folder}'")

# ─────────────────────────────────────────────────────────────────────────────
#  PADDING HELPER
# ─────────────────────────────────────────────────────────────────────────────

def pad_to_multiple(arr: np.ndarray, multiple: int) -> np.ndarray:
    """
    Pad a 2-D array so both dimensions are multiples of `multiple`.
    Uses edge-replication padding.
    """
    h, w = arr.shape
    ph = (multiple - h % multiple) % multiple
    pw = (multiple - w % multiple) % multiple
    return np.pad(arr, ((0, ph), (0, pw)), mode='edge')


def crop(arr: np.ndarray, shape: tuple) -> np.ndarray:
    """Crop padded array back to original shape (h, w)."""
    return arr[:shape[0], :shape[1]]

# ─────────────────────────────────────────────────────────────────────────────
#  COLOUR CONVERSION HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def bgr_to_rgb(frame: np.ndarray) -> np.ndarray:
    """BGR → RGB for matplotlib display."""
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def ycbcr_to_bgr(Y: np.ndarray, Cb: np.ndarray, Cr: np.ndarray) -> np.ndarray:
    """
    Upsample Cb / Cr (4:2:0 → 4:4:4) and convert YCbCr → BGR uint8.
    """
    h, w = Y.shape
    Cb_up = np.repeat(np.repeat(Cb, 2, axis=0), 2, axis=1)[:h, :w]
    Cr_up = np.repeat(np.repeat(Cr, 2, axis=0), 2, axis=1)[:h, :w]

    Cb_s = Cb_up.astype(np.float32) - 128.0
    Cr_s = Cr_up.astype(np.float32) - 128.0

    R = np.clip(Y + 1.402   * Cr_s,             0, 255)
    G = np.clip(Y - 0.34414 * Cb_s - 0.71414 * Cr_s, 0, 255)
    B = np.clip(Y + 1.772   * Cb_s,             0, 255)

    return np.stack([B, G, R], axis=-1).astype(np.uint8)
