"""
entropy_coding.py — Part 4: Entropy Coding (Lossless Compression)
Serialise encoded frame data and compress with zlib (DEFLATE).
Simplified MPEG-4 Video Encoder Pipeline
"""

import pickle
import zlib
import gzip
import io
from pathlib import Path

# Magic header to identify our binary format
_MAGIC = b'MPEG4SIM\x01\x00'

# ─────────────────────────────────────────────────────────────────────────────
#  SERIALISATION
# ─────────────────────────────────────────────────────────────────────────────

def _serialise(encoded_frames: list[dict]) -> bytes:
    """
    Serialise the list of encoded frame dicts with pickle (protocol 5).
    Pickle handles numpy arrays natively and efficiently.
    """
    return pickle.dumps(encoded_frames, protocol=5)


def _deserialise(raw_bytes: bytes) -> list[dict]:
    """Deserialise a pickled byte string back to a list of frame dicts."""
    return pickle.loads(raw_bytes)

# ─────────────────────────────────────────────────────────────────────────────
#  LOSSLESS COMPRESSION  (zlib  /  gzip)
# ─────────────────────────────────────────────────────────────────────────────

def _compress_zlib(data: bytes, level: int = 9) -> bytes:
    """
    Compress with zlib (DEFLATE).
    Level 9 = maximum compression (slowest).
    """
    return zlib.compress(data, level=level)


def _decompress_zlib(data: bytes) -> bytes:
    """Decompress zlib-compressed bytes."""
    return zlib.decompress(data)


def _compress_gzip(data: bytes, level: int = 9) -> bytes:
    """
    Compress with gzip (same DEFLATE algorithm, slightly larger header).
    Kept as an alternative for compatibility.
    """
    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode='wb', compresslevel=level) as f:
        f.write(data)
    return buf.getvalue()


def _decompress_gzip(data: bytes) -> bytes:
    """Decompress gzip-compressed bytes."""
    buf = io.BytesIO(data)
    with gzip.GzipFile(fileobj=buf, mode='rb') as f:
        return f.read()

# ─────────────────────────────────────────────────────────────────────────────
#  PUBLIC API
# ─────────────────────────────────────────────────────────────────────────────

def encode_bitstream(encoded_frames: list[dict],
                     method: str = 'zlib') -> bytes:
    """
    Serialise + compress the encoded frame list.

    Args:
        encoded_frames — output of inter_coding.encode_gop()
        method         — 'zlib' (default) or 'gzip'

    Returns:
        Compressed byte string ready to write to disk.
    """
    raw = _serialise(encoded_frames)

    if method == 'zlib':
        compressed = _compress_zlib(raw)
    elif method == 'gzip':
        compressed = _compress_gzip(raw)
    else:
        raise ValueError(f"Unknown method '{method}'. Choose 'zlib' or 'gzip'.")

    ratio = len(raw) / max(len(compressed), 1)
    print(f"[ENTROPY] Serialised : {len(raw) / 1024:.1f} KB")
    print(f"[ENTROPY] Compressed : {len(compressed) / 1024:.1f} KB  "
          f"(method={method}, ratio={ratio:.2f}×)")
    return compressed


def decode_bitstream(compressed: bytes, method: str = 'zlib') -> list[dict]:
    """
    Decompress + deserialise a byte string back to encoded frame dicts.

    Args:
        compressed — byte string read from .bin file
        method     — must match the method used during encoding

    Returns:
        List of encoded frame dicts.
    """
    if method == 'zlib':
        raw = _decompress_zlib(compressed)
    elif method == 'gzip':
        raw = _decompress_gzip(compressed)
    else:
        raise ValueError(f"Unknown method '{method}'.")

    return _deserialise(raw)


# ─────────────────────────────────────────────────────────────────────────────
#  FILE I/O
# ─────────────────────────────────────────────────────────────────────────────

def write_bin(compressed: bytes, path: Path, method: str = 'zlib'):
    """
    Write compressed bitstream to a .bin file with a magic header.

    File format:
        [0:10]  — magic bytes  b'MPEG4SIM\\x01\\x00'
        [10:11] — method flag  b'Z' (zlib) or b'G' (gzip)
        [11:]   — compressed payload
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    method_flag = b'Z' if method == 'zlib' else b'G'

    with open(path, 'wb') as f:
        f.write(_MAGIC)
        f.write(method_flag)
        f.write(compressed)

    print(f"[ENTROPY] Written → '{path}'  ({path.stat().st_size / 1024:.1f} KB)")


def read_bin(path: Path) -> tuple[bytes, str]:
    """
    Read a .bin file produced by write_bin().

    Returns:
        compressed  — raw compressed payload bytes
        method      — 'zlib' or 'gzip'
    """
    path = Path(path)
    with open(path, 'rb') as f:
        magic = f.read(len(_MAGIC))
        if magic != _MAGIC:
            raise ValueError(f"'{path}' is not a valid MPEG4SIM file "
                             f"(bad magic: {magic!r})")
        method_flag = f.read(1)
        compressed  = f.read()

    method = 'zlib' if method_flag == b'Z' else 'gzip'
    return compressed, method


# ─────────────────────────────────────────────────────────────────────────────
#  CONVENIENCE: FULL ENCODE / DECODE SHORTCUTS
# ─────────────────────────────────────────────────────────────────────────────

def save_encoded(encoded_frames: list[dict],
                 path: Path,
                 method: str = 'zlib') -> int:
    """
    Encode + save in one call.  Returns compressed file size in bytes.
    """
    compressed = encode_bitstream(encoded_frames, method)
    write_bin(compressed, path, method)
    return Path(path).stat().st_size


def load_encoded(path: Path) -> list[dict]:
    """
    Load + decode in one call.  Returns list of encoded frame dicts.
    """
    compressed, method = read_bin(path)
    return decode_bitstream(compressed, method)
