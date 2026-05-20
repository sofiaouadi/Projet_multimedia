"""
visualization.py — Part 5b: Pipeline Visualisation
Single comprehensive matplotlib figure covering all pipeline stages.
Simplified MPEG-4 Video Encoder Pipeline
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from pathlib import Path

from intra_coding  import get_block_stages, QUANT_MATRIX_LUMA
from inter_coding  import compute_residual_map, MACROBLOCK_SIZE
from utils         import bgr_to_rgb

# ─────────────────────────────────────────────────────────────────────────────
#  THEME
# ─────────────────────────────────────────────────────────────────────────────

BG       = '#0d1117'
PANEL    = '#161b22'
BORDER   = '#30363d'
TEXT     = '#e6edf3'
MUTED    = '#8b949e'
RED      = '#ff7b72'
BLUE     = '#79c0ff'
GREEN    = '#56d364'
YELLOW   = '#e3b341'
PURPLE   = '#d2a8ff'

plt.rcParams.update({
    'font.family':  'DejaVu Sans',
    'font.size':    8,
    'text.color':   TEXT,
    'axes.labelcolor': MUTED,
    'xtick.color':  MUTED,
    'ytick.color':  MUTED,
})


def _style(ax, title='', xlabel='', ylabel=''):
    ax.set_facecolor(PANEL)
    for spine in ax.spines.values():
        spine.set_edgecolor(BORDER)
    ax.tick_params(colors=MUTED, labelsize=6)
    if title:
        ax.set_title(title, color=TEXT, fontsize=7.5, pad=4, fontweight='bold')
    if xlabel:
        ax.set_xlabel(xlabel, color=MUTED, fontsize=7)
    if ylabel:
        ax.set_ylabel(ylabel, color=MUTED, fontsize=7)


def _section_banner(fig, text, y_norm):
    """Draw a faint horizontal section label."""
    fig.text(0.5, y_norm, text, ha='center', va='bottom',
             color=PURPLE, fontsize=9, fontweight='bold',
             alpha=0.85)


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN PIPELINE FIGURE
# ─────────────────────────────────────────────────────────────────────────────

def visualise_pipeline(original_frames:  list[np.ndarray],
                        recon_frames:    list[np.ndarray],
                        encoded_frames:  list[dict],
                        preprocessed:   list[dict],
                        ref_Y_list:      list[np.ndarray],
                        metrics:         dict,
                        save_path:       Path = Path('pipeline_visualisation.png'),
                        qf:              float = 1.0):
    """
    Generate a single large matplotlib figure containing all pipeline stages:

        Row 0  — Section banner
        Row 1  — ① Original frames strip  (6 thumbnails)
        Row 2  — ② Y / Cb / Cr channels
        Row 3  — ③ DCT block pipeline (4 sub-plots)
        Row 4  — ④ Motion vectors + ⑤ Residuals & Reconstruction
        Row 5  — ⑥ PSNR per frame  +  Compression summary
        Row 6  — Quantisation matrix + Reconstructed frames strip
    """
    fig = plt.figure(figsize=(22, 26), facecolor=BG)
    fig.suptitle('Simplified MPEG-4 Encoder — Full Pipeline Visualisation',
                 fontsize=17, color=TEXT, fontweight='bold', y=0.995)

    n = len(original_frames)
    strip_idx = np.linspace(0, n - 1, 6, dtype=int)

    # ──────────────────────────────────────────────
    outer = gridspec.GridSpec(7, 1, figure=fig,
                              top=0.975, bottom=0.02,
                              hspace=0.55, left=0.04, right=0.97)

    # ── ROW 1 : Original frames strip ─────────────
    _section_banner(fig, '① Original Frames', 0.963)
    inner1 = gridspec.GridSpecFromSubplotSpec(
        1, 6, subplot_spec=outer[0], wspace=0.06)

    for k, fi in enumerate(strip_idx):
        ax = fig.add_subplot(inner1[0, k])
        ax.imshow(bgr_to_rgb(original_frames[fi]))
        ftype = encoded_frames[fi]['type']
        col   = RED if ftype == 'I' else BLUE
        ax.set_title(f'#{fi} [{ftype}]', color=col, fontsize=7, pad=2)
        ax.axis('off')
        ax.set_facecolor(PANEL)

    # ── ROW 2 : Color channels ─────────────────────
    _section_banner(fig, '② YCbCr Color Space  (4:2:0 Chroma Subsampling)', 0.828)
    inner2 = gridspec.GridSpecFromSubplotSpec(
        1, 3, subplot_spec=outer[1], wspace=0.08)

    pp0 = preprocessed[0]
    ax_y  = fig.add_subplot(inner2[0, 0])
    ax_cb = fig.add_subplot(inner2[0, 1])
    ax_cr = fig.add_subplot(inner2[0, 2])

    ax_y.imshow(pp0['Y'],      cmap='gray',   vmin=0, vmax=255)
    ax_cb.imshow(pp0['Cb_sub'], cmap='Blues_r', vmin=0, vmax=255)
    ax_cr.imshow(pp0['Cr_sub'], cmap='Reds_r',  vmin=0, vmax=255)

    _style(ax_y,  title=f'Y — Luma  (full res {pp0["Y"].shape[1]}×{pp0["Y"].shape[0]})')
    _style(ax_cb, title=f'Cb — Blue-diff  (subsampled ÷2)')
    _style(ax_cr, title=f'Cr — Red-diff   (subsampled ÷2)')
    for ax in [ax_y, ax_cb, ax_cr]:
        ax.axis('off')

    # ── ROW 3 : DCT block pipeline ─────────────────
    _section_banner(fig, '③ DCT + Quantisation  (one 8×8 block)', 0.694)
    inner3 = gridspec.GridSpecFromSubplotSpec(
        1, 4, subplot_spec=outer[2], wspace=0.12)

    stages = get_block_stages(pp0['Y'], block_row=2, block_col=3, qf=qf)
    stage_cfg = [
        ('raw',           'gray',   'Raw pixels\n(8×8 block)'),
        ('dct',           'plasma', 'DCT coefficients\n(log |coeff|)'),
        ('quantised',     'RdBu',   f'Quantised  (QF={qf})\n(many zeros → compact)'),
        ('reconstructed', 'gray',   'Reconstructed block\n(IDCT + dequantise)'),
    ]

    for k, (key, cmap, title) in enumerate(stage_cfg):
        ax  = fig.add_subplot(inner3[0, k])
        dat = stages[key]
        if key == 'dct':
            dat = np.log1p(np.abs(dat))
        im  = ax.imshow(dat, cmap=cmap, interpolation='nearest')
        cb  = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.ax.tick_params(labelsize=5, colors=MUTED)
        _style(ax, title=title)
        ax.set_xticks(range(8))
        ax.set_yticks(range(8))
        ax.tick_params(length=0, labelsize=4, colors=MUTED)

    # ── ROW 4 : Motion vectors + Residuals ─────────
    _section_banner(fig, '④ Motion Vectors          ⑤ Residuals & Reconstruction', 0.558)
    inner4 = gridspec.GridSpecFromSubplotSpec(
        1, 2, subplot_spec=outer[3], wspace=0.08)

    # ─ Motion vectors ─
    ax_mv = fig.add_subplot(inner4[0, 0])
    p_idx = next((i for i, d in enumerate(encoded_frames) if d['type'] == 'P'), None)

    if p_idx is not None:
        ax_mv.imshow(bgr_to_rgb(recon_frames[p_idx]), alpha=0.8)
        mv    = encoded_frames[p_idx]['mv'].astype(float)
        MB    = MACROBLOCK_SIZE
        n_my, n_mx = mv.shape[:2]
        for i in range(n_my):
            for j in range(n_mx):
                dy, dx = mv[i, j, 0], mv[i, j, 1]
                cy = i * MB + MB // 2
                cx = j * MB + MB // 2
                if abs(dy) + abs(dx) > 0:
                    ax_mv.annotate(
                        '', xy=(cx + dx, cy + dy), xytext=(cx, cy),
                        arrowprops=dict(arrowstyle='->', color=YELLOW,
                                        lw=0.9, mutation_scale=8))
        _style(ax_mv, title=f'Motion Vectors — Frame {p_idx} (P-frame)')
    else:
        ax_mv.text(0.5, 0.5, 'No P-frames in this clip',
                   ha='center', va='center', color=MUTED, fontsize=10,
                   transform=ax_mv.transAxes)
        _style(ax_mv, title='Motion Vectors')
    ax_mv.axis('off')

    # ─ Residuals & Reconstruction ─
    inner4b = gridspec.GridSpecFromSubplotSpec(
        1, 3, subplot_spec=inner4[0, 1], wspace=0.06)

    show_idx = p_idx if p_idx is not None else 0
    orig_rgb  = bgr_to_rgb(original_frames[show_idx])
    recon_rgb = bgr_to_rgb(recon_frames[show_idx])
    res_map   = compute_residual_map(
        pp0['Y'],  # use luma channel of frame 0 for illustration
        ref_Y_list[show_idx])

    for k, (img, cmap, title) in enumerate([
        (orig_rgb,  None,  f'Original #{show_idx}'),
        (res_map,   'hot', 'Residual map\n(|orig − recon|)'),
        (recon_rgb, None,  f'Reconstructed #{show_idx}'),
    ]):
        ax = fig.add_subplot(inner4b[0, k])
        ax.imshow(img, cmap=cmap)
        _style(ax, title=title)
        ax.axis('off')

    # ── ROW 5 : PSNR chart + Stats ─────────────────
    _section_banner(fig, '⑥ Quality Metrics & Compression Statistics', 0.420)
    inner5 = gridspec.GridSpecFromSubplotSpec(
        1, 2, subplot_spec=outer[4], wspace=0.12, width_ratios=[3, 1])

    ax_psnr = fig.add_subplot(inner5[0, 0])
    colors  = [RED if t == 'I' else BLUE for t in metrics['frame_types']]
    ax_psnr.bar(range(n), metrics['psnr'], color=colors, width=0.85, zorder=2)
    ax_psnr.axhline(metrics['avg_psnr'], color=YELLOW, lw=1.4,
                    linestyle='--', zorder=3,
                    label=f"Avg PSNR = {metrics['avg_psnr']:.2f} dB")
    ax_psnr.grid(axis='y', color=BORDER, linewidth=0.5, zorder=1)
    _style(ax_psnr, title='PSNR per Frame (dB)',
           xlabel='Frame index', ylabel='PSNR (dB)')

    i_patch = mpatches.Patch(color=RED,  label='I-frame')
    p_patch = mpatches.Patch(color=BLUE, label='P-frame')
    ax_psnr.legend(handles=[i_patch, p_patch,
                             plt.Line2D([0], [0], color=YELLOW, lw=1.5,
                                        linestyle='--', label=f"Avg PSNR={metrics['avg_psnr']:.1f} dB")],
                   fontsize=7, labelcolor=TEXT,
                   facecolor=PANEL, edgecolor=BORDER)

    ax_stat = fig.add_subplot(inner5[0, 1])
    _style(ax_stat, title='Compression Summary')
    ax_stat.axis('off')
    lines = [
        f"Frames        {metrics['n_frames']}",
        f"I-frames      {metrics['n_iframes']}",
        f"P-frames      {metrics['n_pframes']}",
        f"",
        f"Original      {metrics['original_bytes']/1024:.1f} KB",
        f"Compressed    {metrics['compressed_bytes']/1024:.1f} KB",
        f"Ratio         {metrics['compression_ratio']:.2f}×",
        f"",
        f"Avg PSNR      {metrics['avg_psnr']:.2f} dB",
        f"Avg SSIM      {metrics['avg_ssim']:.4f}",
        f"Avg MSE       {metrics['avg_mse']:.2f}",
    ]
    for k, line in enumerate(lines):
        ax_stat.text(0.06, 0.96 - k * 0.083, line,
                     color=TEXT, fontsize=8,
                     transform=ax_stat.transAxes,
                     fontfamily='monospace')

    # ── ROW 6 : Quantisation matrix + Recon strip ──
    _section_banner(fig, '⑦ Quantisation Matrix   |   Reconstructed Frame Strip', 0.275)
    inner6 = gridspec.GridSpecFromSubplotSpec(
        1, 7, subplot_spec=outer[5], wspace=0.06, width_ratios=[2,1,1,1,1,1,1])

    ax_qm = fig.add_subplot(inner6[0, 0])
    im_qm = ax_qm.imshow(QUANT_MATRIX_LUMA, cmap='YlOrRd', interpolation='nearest')
    for i in range(8):
        for j in range(8):
            ax_qm.text(j, i, int(QUANT_MATRIX_LUMA[i, j]),
                       ha='center', va='center', fontsize=5, color='black',
                       fontweight='bold')
    plt.colorbar(im_qm, ax=ax_qm, fraction=0.046).ax.tick_params(
        labelsize=5, colors=MUTED)
    _style(ax_qm, title='Quantisation Matrix\n(JPEG luma standard)')
    ax_qm.set_xticks(range(8)); ax_qm.set_yticks(range(8))
    ax_qm.tick_params(length=0, labelsize=4, colors=MUTED)

    for k, fi in enumerate(strip_idx):
        ax = fig.add_subplot(inner6[0, k + 1])
        ax.imshow(bgr_to_rgb(recon_frames[fi]))
        ax.set_title(f'Recon #{fi}', color=GREEN, fontsize=6, pad=2)
        ax.axis('off')
        ax.set_facecolor(PANEL)

    # ── ROW 7 : SSIM bar chart ─────────────────────
    _section_banner(fig, '⑧ SSIM per Frame', 0.135)
    inner7 = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=outer[6])
    ax_ssim = fig.add_subplot(inner7[0, 0])
    ax_ssim.bar(range(n), metrics['ssim'], color=GREEN, alpha=0.8, width=0.85)
    ax_ssim.axhline(metrics['avg_ssim'], color=YELLOW, lw=1.2, linestyle='--',
                    label=f"Avg SSIM = {metrics['avg_ssim']:.4f}")
    ax_ssim.set_ylim(max(0, min(metrics['ssim']) - 0.05), 1.02)
    ax_ssim.grid(axis='y', color=BORDER, linewidth=0.5)
    _style(ax_ssim, title='SSIM per Frame',
           xlabel='Frame index', ylabel='SSIM')
    ax_ssim.legend(fontsize=7, labelcolor=TEXT,
                   facecolor=PANEL, edgecolor=BORDER)

    plt.savefig(str(save_path), dpi=150,
                bbox_inches='tight', facecolor=BG)
    print(f"[VIZ] Pipeline figure saved → '{save_path}'")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
#  EXPERIMENTAL ANALYSIS FIGURES
# ─────────────────────────────────────────────────────────────────────────────

def plot_qf_analysis(qf_results: dict,
                     save_path: Path = Path('qf_analysis.png')):
    """
    Three sub-plots: compression ratio, PSNR, and SSIM vs QF.
    """
    qfs    = sorted(qf_results.keys())
    ratios = [qf_results[q]['ratio']    for q in qfs]
    psnrs  = [qf_results[q]['avg_psnr'] for q in qfs]
    ssims  = [qf_results[q]['avg_ssim'] for q in qfs]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), facecolor=BG)
    fig.suptitle('Effect of Quantisation Factor (QF) on Compression vs Quality',
                 color=TEXT, fontsize=13, fontweight='bold')

    for ax, y, ylabel, color, title in zip(
            axes,
            [ratios, psnrs, ssims],
            ['Compression Ratio (×)', 'Avg PSNR (dB)', 'Avg SSIM'],
            [BLUE, RED, GREEN],
            ['Compression Ratio vs QF',
             'PSNR vs QF  (quality)',
             'SSIM vs QF  (perceptual quality)']):
        ax.set_facecolor(PANEL)
        ax.plot(qfs, y, marker='o', color=color, linewidth=2,
                markersize=7, zorder=3)
        ax.fill_between(qfs, y, alpha=0.15, color=color)
        ax.grid(color=BORDER, linewidth=0.6, zorder=1)
        for s in ax.spines.values():
            s.set_edgecolor(BORDER)
        ax.tick_params(colors=MUTED)
        ax.set_xlabel('Quantisation Factor (QF)', color=MUTED, fontsize=9)
        ax.set_ylabel(ylabel, color=MUTED, fontsize=9)
        ax.set_title(title, color=TEXT, fontsize=10, fontweight='bold')
        # Annotate each point
        for xi, yi in zip(qfs, y):
            ax.annotate(f'{yi:.1f}', (xi, yi),
                        textcoords='offset points', xytext=(0, 7),
                        ha='center', fontsize=7, color=color)

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(str(save_path), dpi=140,
                bbox_inches='tight', facecolor=BG)
    print(f"[VIZ] QF analysis → '{save_path}'")


def plot_gop_analysis(gop_results: dict,
                      save_path: Path = Path('gop_analysis.png')):
    """
    Single plot: compression ratio vs GOP size.
    """
    gops   = sorted(gop_results.keys())
    ratios = [gop_results[g] for g in gops]

    fig, ax = plt.subplots(figsize=(9, 5), facecolor=BG)
    ax.set_facecolor(PANEL)
    ax.plot(gops, ratios, marker='s', color=PURPLE,
            linewidth=2.5, markersize=9, zorder=3)
    ax.fill_between(gops, ratios, alpha=0.15, color=PURPLE)
    ax.grid(color=BORDER, linewidth=0.6, zorder=1)
    for s in ax.spines.values():
        s.set_edgecolor(BORDER)
    ax.tick_params(colors=MUTED)
    ax.set_xlabel('GOP Size (G)', color=MUTED, fontsize=11)
    ax.set_ylabel('Compression Ratio (×)', color=MUTED, fontsize=11)
    ax.set_title('Effect of GOP Size on Compression Ratio',
                 color=TEXT, fontsize=13, fontweight='bold')

    for xi, yi in zip(gops, ratios):
        ax.annotate(f'{yi:.2f}×', (xi, yi),
                    textcoords='offset points', xytext=(0, 9),
                    ha='center', fontsize=8, color=PURPLE)

    ax.set_xticks(gops)
    plt.tight_layout()
    plt.savefig(str(save_path), dpi=140,
                bbox_inches='tight', facecolor=BG)
    print(f"[VIZ] GOP analysis → '{save_path}'")
