"""
main.py — Unified CLI Entry Point
Simplified MPEG-4 Video Encoder Pipeline
Multimedia Systems Project 2026

Usage:
    python main.py encode   [--frames DIR] [--output FILE] [--gop N] [--qf F] [--search N]
    python main.py decode   [--bin FILE]   [--decoded DIR]
    python main.py full     [--frames DIR] [--output FILE] [--gop N] [--qf F] [--search N]
    python main.py analyse  [--frames DIR] [--gop N] [--qf F]
"""

import sys
import argparse
from pathlib import Path
import numpy as np

# ── ensure src/ is on the import path when called from project root ──────────
sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils         import load_frames, save_frames, ensure_dirs, OUTPUT_DIR, INPUT_DIR
from preprocessing import preprocess_all
from inter_coding  import encode_gop
from entropy_coding import save_encoded
from decoder       import decode_video
from metrics       import compute_all_metrics, sweep_qf, sweep_gop
from visualization import (visualise_pipeline,
                            plot_qf_analysis,
                            plot_gop_analysis)


# ─────────────────────────────────────────────────────────────────────────────
#  HELPER : generate synthetic test frames if input folder is empty
# ─────────────────────────────────────────────────────────────────────────────

def _generate_test_frames(folder: Path, n: int = 30,
                           w: int = 128, h: int = 96):
    """Create n synthetic BGR frames (animated gradient + moving circle)."""
    import cv2
    folder.mkdir(parents=True, exist_ok=True)
    existing = list(folder.glob('*.png')) + list(folder.glob('*.jpg'))
    if existing:
        return  # already populated

    print(f"[MAIN] No frames found — generating {n} synthetic test frames in '{folder}'")
    for i in range(n):
        t   = i / n
        img = np.zeros((h, w, 3), dtype=np.uint8)
        # Animated colour gradient background
        for y in range(h):
            for x in range(w):
                img[y, x, 0] = int(255 * ((x / w + t * 0.3) % 1.0))   # B
                img[y, x, 1] = int(255 * ((y / h + t * 0.2) % 1.0))   # G
                img[y, x, 2] = int(255 * t)                             # R
        # Moving white circle
        cx  = int(w * 0.1 + w * 0.8 * t)
        cy  = h // 2
        r   = 10
        cv2.circle(img, (cx, cy), r, (200, 230, 255), -1)
        # Moving rectangle
        rx  = int(w * 0.9 - w * 0.8 * t)
        cv2.rectangle(img, (rx, 8), (rx + 18, 28), (100, 200, 120), -1)

        cv2.imwrite(str(folder / f'frame_{i:04d}.png'), img)

    print(f"[MAIN] Generated {n} frames  ({w}×{h} px)")


# ─────────────────────────────────────────────────────────────────────────────
#  COMMANDS
# ─────────────────────────────────────────────────────────────────────────────

def cmd_encode(args):
    """Part 1-4: Pre-process → encode GOP → entropy code → write .bin"""
    ensure_dirs()
    frames_dir = Path(args.frames)
    _generate_test_frames(frames_dir)

    frames     = load_frames(frames_dir)
    pp_frames  = preprocess_all(frames)
    enc_frames, _ = encode_gop(pp_frames, qf=args.qf,
                                gop=args.gop, search=args.search)
    out = Path(args.output)
    save_encoded(enc_frames, out)
    print(f"\n✅  Encoded → '{out}'")


def cmd_decode(args):
    """Part 4 decoder: read .bin → reconstruct frames → save PNGs."""
    ensure_dirs()
    recon = decode_video(Path(args.bin_file), Path(args.decoded))
    print(f"\n✅  Decoded {len(recon)} frames → '{args.decoded}'")


def cmd_full(args):
    """Full pipeline: encode + decode + metrics + visualise."""
    ensure_dirs()
    frames_dir = Path(args.frames)
    _generate_test_frames(frames_dir)

    out_dir  = OUTPUT_DIR
    bin_path = Path(args.output)
    dec_dir  = Path(args.decoded)

    # ── Encode ──────────────────────────────────────────────────────
    print("\n" + "═"*55)
    print("  STAGE 1–4 : ENCODING")
    print("═"*55)

    frames    = load_frames(frames_dir)
    pp_frames = preprocess_all(frames)

    enc_frames, ref_Y_list = encode_gop(
        pp_frames, qf=args.qf, gop=args.gop, search=args.search)

    save_encoded(enc_frames, bin_path)

    # ── Decode ──────────────────────────────────────────────────────
    print("\n" + "═"*55)
    print("  STAGE 4   : DECODING")
    print("═"*55)

    recon_frames = decode_video(bin_path, dec_dir)

    # ── Metrics ─────────────────────────────────────────────────────
    print("\n" + "═"*55)
    print("  STAGE 5a  : METRICS")
    print("═"*55)

    metrics = compute_all_metrics(
        frames, recon_frames, enc_frames, bin_path)

    # ── Visualise ───────────────────────────────────────────────────
    print("\n" + "═"*55)
    print("  STAGE 5b  : VISUALISATION")
    print("═"*55)

    viz_path = out_dir / 'pipeline_visualisation.png'
    visualise_pipeline(
        original_frames = frames,
        recon_frames    = recon_frames,
        encoded_frames  = enc_frames,
        preprocessed    = pp_frames,
        ref_Y_list      = ref_Y_list,
        metrics         = metrics,
        save_path       = viz_path,
        qf              = args.qf,
    )

    # ── Summary ─────────────────────────────────────────────────────
    print("\n" + "═"*55)
    print("  ✅  PIPELINE COMPLETE")
    print("═"*55)
    print(f"  Encoded file      : {bin_path}")
    print(f"  Decoded frames    : {dec_dir}/")
    print(f"  Visualisation     : {viz_path}")
    print(f"  Compression ratio : {metrics['compression_ratio']:.2f}×")
    print(f"  Avg PSNR          : {metrics['avg_psnr']:.2f} dB")
    print(f"  Avg SSIM          : {metrics['avg_ssim']:.4f}")
    print("═"*55 + "\n")


def cmd_analyse(args):
    """Experimental analysis: QF sweep + GOP sweep → plots."""
    ensure_dirs()
    frames_dir = Path(args.frames)
    _generate_test_frames(frames_dir)

    out_dir = OUTPUT_DIR
    frames  = load_frames(frames_dir)

    # ── QF sweep ────────────────────────────────────────────────────
    print("\n" + "═"*55)
    print("  ANALYSIS : Quantisation Factor Sweep")
    print("═"*55)

    qf_values  = [0.5, 1.0, 2.0, 4.0, 8.0, 16.0]
    qf_results = sweep_qf(frames, qf_values,
                           gop=args.gop, search=4,
                           tmp_bin=Path('/tmp/_qf_sweep.bin'))
    plot_qf_analysis(qf_results,
                     save_path=out_dir / 'qf_analysis.png')

    # ── GOP sweep ───────────────────────────────────────────────────
    print("\n" + "═"*55)
    print("  ANALYSIS : GOP Size Sweep")
    print("═"*55)

    n_frames  = len(frames)
    gop_vals  = sorted(set([1, 2, 5, 10, 15, 20, n_frames]))
    gop_vals  = [g for g in gop_vals if g <= n_frames]
    gop_results = sweep_gop(frames, gop_vals,
                             qf=args.qf, search=4,
                             tmp_bin=Path('/tmp/_gop_sweep.bin'))
    plot_gop_analysis(gop_results,
                      save_path=out_dir / 'gop_analysis.png')

    print("\n✅  Analysis done!")
    print(f"   {out_dir / 'qf_analysis.png'}")
    print(f"   {out_dir / 'gop_analysis.png'}\n")


# ─────────────────────────────────────────────────────────────────────────────
#  ARGUMENT PARSER
# ─────────────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog='main.py',
        description='Simplified MPEG-4 Video Encoder — Multimedia Systems 2026',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py full                         # run everything with defaults
  python main.py full    --gop 5  --qf 2.0   # higher compression
  python main.py encode  --frames data/input_frames --output data/output/video.bin
  python main.py decode  --bin data/output/video.bin --decoded data/output/decoded/
  python main.py analyse --gop 10 --qf 1.0
        """)

    sub = parser.add_subparsers(dest='command', metavar='COMMAND')

    def _add_common(p):
        p.add_argument('--frames',  default=str(INPUT_DIR),
                       help=f'Input frames folder  (default: {INPUT_DIR})')
        p.add_argument('--output',  default=str(OUTPUT_DIR / 'video.bin'),
                       help=f'Output .bin file  (default: data/output/video.bin)')
        p.add_argument('--decoded', default=str(OUTPUT_DIR / 'decoded'),
                       help='Output folder for decoded frames  (default: data/output/decoded)')
        p.add_argument('--gop',    type=int,   default=10,
                       help='GOP size — every N-th frame is an I-frame  (default: 10)')
        p.add_argument('--qf',     type=float, default=1.0,
                       help='Quantisation factor  (default: 1.0; higher=more compression)')
        p.add_argument('--search', type=int,   default=8,
                       help='Motion search window ±S pixels  (default: 8)')

    # encode
    p_enc = sub.add_parser('encode', help='Encode input frames → .bin')
    _add_common(p_enc)

    # decode
    p_dec = sub.add_parser('decode', help='Decode .bin → reconstructed frames')
    p_dec.add_argument('--bin-file', dest='bin_file',
                       default=str(OUTPUT_DIR / 'video.bin'),
                       help='Input .bin file')
    p_dec.add_argument('--decoded',  default=str(OUTPUT_DIR / 'decoded'),
                       help='Output folder for decoded frames')
    p_dec.add_argument('--qf', type=float, default=1.0)

    # full
    p_full = sub.add_parser('full', help='Full pipeline: encode + decode + metrics + visualise')
    _add_common(p_full)

    # analyse
    p_an = sub.add_parser('analyse', help='Run QF + GOP experimental sweeps')
    _add_common(p_an)

    return parser


def main():
    parser = _build_parser()
    args   = parser.parse_args()

    if args.command == 'encode':
        cmd_encode(args)
    elif args.command == 'decode':
        cmd_decode(args)
    elif args.command == 'full':
        cmd_full(args)
    elif args.command == 'analyse':
        cmd_analyse(args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
