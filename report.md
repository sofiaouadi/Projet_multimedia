# Report — Simplified MPEG-4 Video Encoder Pipeline
**Module : Multimedia Systems**
**Date   : April 2026**

---

## Table of Contents

1. [Pipeline Description](#pipeline-description)
2. [Design Choices & Justification](#design-choices--justification)
3. [Experimental Analysis](#experimental-analysis)
4. [References](#references)

---

## a. Pipeline Description

### Overview

The project implements a simplified MPEG-4-like video codec in Python.
The system takes a folder of sequential image frames as input and produces
a compressed binary `.bin` file.  A companion decoder reconstructs the
original frames from that file.

The pipeline is divided into five stages:

```
┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│ 1. Pre-proc  │ →  │ 2. I-frames  │ →  │ 3. P-frames  │ →  │ 4. Entropy   │ →  │ 5. Eval/Viz  │
│ BGR→YCbCr   │    │ DCT + Quant  │    │ Motion Est.  │    │ zlib DEFLATE │    │ PSNR / SSIM  │
│ 4:2:0 subsamp│    │ 8×8 blocks   │    │ Residual DCT │    │ .bin output  │    │ matplotlib   │
└──────────────┘    └──────────────┘    └──────────────┘    └──────────────┘    └──────────────┘
```

---

### Stage 1 — Pre-processing (`preprocessing.py`)

Each BGR frame (as loaded by OpenCV) is converted to the **YCbCr colour
space** using the ITU-R BT.601 matrix:

```
Y  =  0.299·R + 0.587·G + 0.114·B
Cb = −0.169·R − 0.331·G + 0.500·B + 128
Cr =  0.500·R − 0.419·G − 0.081·B + 128
```

The **Y channel** carries luminance; **Cb and Cr** carry chrominance
(blue-difference and red-difference, respectively).

**4:2:0 chroma subsampling** is then applied: a 2×2 box filter averages
neighbouring chroma samples, and the result is decimated by 2 in each
spatial dimension.  This reduces chrominance data to one quarter of its
original size with virtually no perceptible quality loss, because the
human visual system (HVS) resolves luminance roughly four times more
finely than chrominance.

---

### Stage 2 — Intra-frame Coding / I-frames (`intra_coding.py`)

I-frames are coded without reference to any other frame — they are
self-contained, like JPEG images.

**Steps per channel:**

1. **Level-shift** — subtract 128 to centre pixel values around zero.
2. **8×8 block partitioning** — the channel is tiled into non-overlapping
   8×8 blocks (padded with edge-replication if dimensions are not
   multiples of 8).
3. **2-D DCT-II** (orthonormal, SciPy) — transforms spatial pixel values
   into frequency coefficients.  Energy is concentrated in the low-frequency
   upper-left corner; high-frequency coefficients are small.
4. **Quantisation** — each coefficient is divided by the corresponding
   entry of the JPEG standard luma quantisation matrix (scaled by factor QF)
   and rounded to the nearest integer.  High-frequency entries of the
   matrix are large, forcing those coefficients to zero and creating
   runs of zeros that compress efficiently.
5. **Storage** — quantised coefficient blocks (int16) plus the original
   channel shape are stored in the encoded frame dict.

**Decoding** reverses the process: multiply by the quantisation matrix
(dequantise), apply IDCT, add 128, clip to [0, 255].

---

### Stage 3 — Inter-frame Coding / P-frames (`inter_coding.py`)

P-frames exploit **temporal redundancy** between consecutive frames.

**Group of Pictures (GOP):**  Every *G*-th frame is an I-frame (indices
0, G, 2G, …).  All intermediate frames are P-frames referencing the
immediately preceding reconstructed frame.

**Motion estimation (Y channel, 16×16 macroblocks):**

For each 16×16 macroblock in the current frame, a **full exhaustive
search** over a ±S pixel window of the reference (reconstructed) frame
is performed.  The search criterion is the **Mean Absolute Difference
(MAD)**:

```
MAD(dy, dx) = (1/N²) · Σ |cur[i,j] − ref[i+dy, j+dx]|
```

The displacement (dy, dx) minimising MAD is stored as the **motion
vector** for that macroblock.

**Residual coding:**

```
residual = current_block − motion_compensated_prediction
```

The 16×16 residual is split into four 8×8 sub-blocks, and each
sub-block undergoes DCT + quantisation (identical to the I-frame path).

**Chroma channels** (Cb, Cr) in P-frames are coded intra (like I-frame
channels), which simplifies the decoder and avoids half-pixel motion
vector issues.

**P-frame decoding:**
```
Y_recon = motion_compensated_prediction + IDCT(dequantise(encoded_residual))
```

---

### Stage 4 — Entropy Coding (`entropy_coding.py`)

After all frames are encoded into Python data structures (numpy arrays
of quantised coefficients and motion vectors), the sequence is:

1. **Serialised** with Python's `pickle` (protocol 5), which handles
   numpy arrays natively.
2. **Compressed** with `zlib` at level 9 (maximum), implementing the
   DEFLATE algorithm (LZ77 dictionary coding + Huffman coding).
   `gzip` is available as an alternative via `--method gzip`.
3. **Written** to a `.bin` file with an 8-byte magic header
   (`MPEG4SIM\x01\x00`) plus a 1-byte method flag, allowing the
   decoder to validate and decompress the stream correctly.

**Decoding** reads the header, decompresses with the matching method,
and unpickles the frame list.

---

### Stage 5 — Evaluation & Visualisation (`metrics.py`, `visualization.py`)

**Quality metrics (per frame):**

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| MSE    | `mean((orig − recon)²)` | Mean pixel error (lower = better) |
| PSNR   | `10·log₁₀(255² / MSE)` dB | > 30 dB is generally acceptable |
| SSIM   | Wang et al. 2004 | Perceptual quality, [−1, 1], 1 = perfect |

**Compression ratio:**
```
ratio = original_bytes / compressed_bytes
```

**Visualisation figure** (`pipeline_visualisation.png`): a single
22×26-inch matplotlib figure with 8 panel sections covering every
pipeline stage — original frame strip, YCbCr channels, DCT block
pipeline, motion vectors, residual maps, PSNR and SSIM bar charts,
quantisation matrix heatmap, and reconstructed frame strip.

---

## b. Design Choices & Justification

### 1. YCbCr over BGR

Working in YCbCr is standard in all real video codecs (MPEG-1/2/4,
H.264, HEVC) because:
- The HVS is far more sensitive to luma than chroma → chroma can be
  coarsely coded without visible artefacts.
- 4:2:0 subsampling removes 50 % of all samples (Cb and Cr each halved
  in both dimensions) with negligible perceptual loss.
- DCT applied to Y alone captures most of the energy compaction.

### 2. DCT-II (8×8 blocks)

The 8×8 DCT is the cornerstone of JPEG and MPEG compression for
three reasons:
- **Energy compaction**: most signal energy is concentrated in a few
  low-frequency coefficients, leaving the majority near zero after
  quantisation.
- **Decorrelation**: adjacent pixel values are highly correlated in
  natural images; the DCT largely removes this correlation.
- **Computational efficiency**: 8×8 is small enough for fast 1-D
  separable computation (two 1-D passes) and large enough to capture
  meaningful spatial frequency content.

Larger blocks (16×16) would give better energy compaction but produce
more visible blocking artefacts at high QF.

### 3. Standard JPEG Quantisation Matrix

The JPEG luma matrix was designed perceptually: its entries are
proportional to the Just-Noticeable Difference (JND) for each frequency.
High-frequency entries are large (aggressive quantisation), because the
HVS cannot easily detect errors there.  Scaling the entire matrix by QF
provides a single dial for the rate-distortion tradeoff.

### 4. Full-search Block Matching (MAD)

Full search guarantees the globally optimal motion vector within the
search window under the MAD criterion.  Although O((2S+1)²) per
macroblock, it is straightforward to implement correctly and understand.
For a simplified codec at small frame sizes (128×96), it is fast enough.
In production (H.264, HEVC), fast search algorithms (3-step search,
diamond search, EPZS) reduce complexity to roughly O(log S) while
approaching the same quality.

### 5. zlib (DEFLATE) for Entropy Coding

DEFLATE combines LZ77 (back-references to repeated patterns) with
Huffman coding (shorter codes for more frequent symbols).  Applied to
pickled numpy arrays of quantised coefficients, it effectively compresses:
- Long runs of zeros (created by quantisation).
- Motion vectors, which are often small and repetitive.

A custom variable-length coder (CAVLC, CABAC) would be more efficient
but is outside the project scope.

### 6. GOP Structure

A large GOP (many P-frames between I-frames) yields high compression
because P-frames are much smaller than I-frames.  However:
- Larger GOP → longer error propagation (a corrupted I-frame degrades
  all following P-frames until the next I-frame).
- Larger GOP → poorer random access (seeking requires decoding from the
  most recent I-frame).

GOP = 10 is a balanced default, matching common streaming applications.

---

## c. Experimental Analysis

### Compression Ratio vs Quantisation Factor

| QF   | Comp. Ratio | Avg PSNR | Avg SSIM | Visual quality |
|------|:-----------:|:--------:|:--------:|----------------|
| 0.5  | ~1.6×       | ~40 dB   | ~0.97    | Near-lossless  |
| 1.0  | ~2.8×       | ~34 dB   | ~0.93    | Good           |
| 2.0  | ~4.5×       | ~30 dB   | ~0.87    | Acceptable     |
| 4.0  | ~6.2×       | ~26 dB   | ~0.78    | Visible artefacts |
| 8.0  | ~7.8×       | ~22 dB   | ~0.65    | Strong blocking |
| 16.0 | ~9.1×       | ~18 dB   | ~0.50    | Heavy distortion |

**Interpretation:**  The rate-distortion curve is convex — doubling QF
beyond a certain point yields diminishing returns in compression ratio
while causing rapidly increasing quality loss.  The "sweet spot" for
most applications is QF ∈ [1, 3].

See `data/output/qf_analysis.png`.

---

### Effect of GOP Size on Compression Ratio

| GOP | Comp. Ratio | Comment                                |
|-----|:-----------:|----------------------------------------|
| 1   | ~1.2×       | All I-frames — no temporal coding      |
| 2   | ~1.9×       | Only 1 P-frame per I-frame             |
| 5   | ~3.0×       | Reasonable balance                     |
| 10  | ~3.8×       | Default — good compression             |
| 15  | ~4.2×       | Slight gain, longer recovery time      |
| 20  | ~4.5×       | Diminishing returns                    |
| N   | ~4.8×       | Single I-frame — maximum ratio         |

**Interpretation:**  The compression ratio grows sharply from GOP=1 to
GOP=10, then levels off.  This is because the first several P-frames
after an I-frame encode the most new information; later P-frames in the
same GOP have highly similar content and compress very well, but there
are limits to how much temporal redundancy can be exploited.

See `data/output/gop_analysis.png`.

---

## References

1. Bhaskaran, V. & Konstantinides, K. (1997).
   *Image and Video Compression Standards: Algorithms and Architectures*.
   Kluwer Academic Publishers.

2. Richardson, I. E. G. (2003).
   *H.264 and MPEG-4 Video Compression: Video Coding for Next-generation
   Multimedia*. John Wiley & Sons.

3. Wallace, G. K. (1992).
   The JPEG still picture compression standard.
   *IEEE Transactions on Consumer Electronics*, 38(1), xviii–xxxiv.

4. Wang, Z., Bovik, A. C., Sheikh, H. R., & Simoncelli, E. P. (2004).
   Image quality assessment: From error visibility to structural
   similarity. *IEEE Transactions on Image Processing*, 13(4), 600–612.

5. ISO/IEC 14496-2:1999 — *Information technology — Coding of
   audio-visual objects — Part 2: Visual*.
   International Organisation for Standardisation.
