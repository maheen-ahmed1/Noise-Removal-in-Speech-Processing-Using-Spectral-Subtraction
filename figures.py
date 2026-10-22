import os

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Ellipse, FancyArrowPatch, Rectangle
from mpl_toolkits.mplot3d import Axes3D

SHOW = False
CAPTIONS = True
BLUE, ORANGE, GREY, CAPTION = "#2a6f8f", "#c46a3f", "#7a7a7a", "#2a6f8f"


# Display Setup

def setup(show=True):
    global SHOW
    SHOW = show
    if not show:
        plt.switch_backend("Agg")
    elif matplotlib.get_backend().lower() == "agg":
        try:
            get_ipython().run_line_magic("matplotlib", "inline")
        except Exception:
            pass


def show():
    if SHOW:
        print("\nShowing the figures (Spyder: Plots pane; terminal: close the "
              "windows to finish; --no-show to skip).")
        plt.show()


def save(fig, outdir, name):
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, name), dpi=200)
    if not SHOW:
        plt.close(fig)


# Plot Helpers

def _caption(fig, text, y=0.005):
    if CAPTIONS:
        fig.suptitle(text, fontsize=9, color=CAPTION, y=y, va="bottom")


def _sci_x(ax):
    ax.ticklabel_format(axis="x", style="sci", scilimits=(4, 4))


def _box(ax, cx, cy, w, h, text, fs=8):
    ax.add_patch(Rectangle((cx - w / 2, cy - h / 2), w, h, fc="white", ec="black", lw=1.2))
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fs)


def _arrow(ax, x0, y0, x1, y1, label=None):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>",
                                 mutation_scale=11, lw=1.1, color="black"))
    if label:
        ax.text((x0 + x1) / 2, max(y0, y1) + 0.16, label, ha="center", fontsize=8)


def _mesh(ax, mag, bins=80):
    m = mag[:, :bins]
    F, T = np.meshgrid(np.arange(m.shape[1]), np.arange(m.shape[0]))
    ax.plot_wireframe(F, T, m, rstride=max(1, len(m) // 70), cstride=1,
                      lw=0.28, color="black")
    ax.set_zlim(0, 40)
    ax.set_xlim(0, m.shape[1])
    ax.set_ylim(0, len(m))
    ax.view_init(elev=32, azim=-56)
    ax.set_xlabel("frequency")
    ax.set_ylabel("time frame")
    ax.set_zlabel("Magnitude of FFT")


# Paper Figures (1 to 6)

def figure_1(outdir):
    fig, ax = plt.subplots(figsize=(11, 3.4))
    ax.set_xlim(0, 11.6)
    ax.set_ylim(0, 3.4)
    ax.axis("off")
    for x, y, t, fs in ((0.55, 2.75, "Speech Signal", 8), (0.55, 2.45, "$S(t)$", 9),
                        (0.55, 0.95, "Noise Signal", 8), (0.55, 1.25, "$N(t)$", 9),
                        (3.35, 0.5, r"$f_s = \frac{1}{T_s}\,Hz$", 8),
                        (7.40, 0.7, "$W[n]$", 8)):
        ax.text(x, y, t, ha="center", fontsize=fs)
    for cx, sym in ((1.85, "+"), (7.4, r"$\times$")):
        ax.add_patch(Circle((cx, 1.85), 0.19, fc="white", ec="black", lw=1.2))
        ax.text(cx, 1.85, sym, ha="center", va="center", fontsize=11)
    _box(ax, 3.35, 1.85, 1.5, 0.8, "A-to-D\nConverter")
    _box(ax, 5.55, 1.85, 1.9, 0.8, "Half-Overlapped\nData Buffer")
    _box(ax, 8.85, 1.85, 1.1, 0.8, "FFT")
    for a in ((1.05, 2.35, 1.72, 1.99), (1.05, 1.35, 1.72, 1.71),
              (2.04, 1.85, 2.6, 1.85, "$X(t)$"), (4.10, 1.85, 4.6, 1.85, "$X[n]$"),
              (6.50, 1.85, 7.21, 1.85, r"$\hat{X}[n]$"),
              (7.59, 1.85, 8.3, 1.85, r"$\hat{X}[n]$"),
              (9.40, 1.85, 10.1, 1.85, r"$\tilde{X}[k]$"),
              (3.35, 0.75, 3.35, 1.45), (7.40, 0.95, 7.40, 1.66)):
        _arrow(ax, *a)
    if CAPTIONS:
        ax.set_title("Figure 1. Block diagram of noisy speech generation and "
                     "discretization.", fontsize=9, color=CAPTION)
    save(fig, outdir, "figure1_block_diagram.png")


def figure_2(speech, noise, noisy, outdir, noise_label="Vacuum cleaner noise"):
    fig, axes = plt.subplots(3, 1, figsize=(7.2, 6.4))
    for ax, sig, lab, tag in zip(axes, (speech, noise, noisy),
                                 ("$S[n]$", "$N[n]$", "$X[n]$"), "abc"):
        ax.plot(sig, lw=0.4, color="black")
        ax.set_xlim(0, len(sig))
        ax.set_ylim(-1, 1)
        ax.set_yticks([-1, 0, 1])
        ax.set_ylabel(lab)
        ax.set_xlabel("Discrete-time n")
        _sci_x(ax)
        ax.text(-0.11, -0.30, "(%s)" % tag, transform=ax.transAxes, fontsize=10)
    _caption(fig, "Figure 2. (a) Clean speech $S[n]$; (b) %s $N[n]$;\n(c) Noisy "
                  "speech corrupted by the noise $X[n]$" % noise_label)
    fig.tight_layout(rect=[0, 0.07, 1, 1])
    save(fig, outdir, "figure2_time_waveforms.png")


def figure_3(outdir):
    fig, ax = plt.subplots(figsize=(8.6, 9.2))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 12.4)
    ax.axis("off")
    for b in ((2.6, 11.5, 3.6, 1.3, "Noisy speech frame\nduring speech activity\n"
               r"$\tilde{X}[k]$"),
              (7.2, 11.5, 3.6, 1.3, "Noise frame during\nnon-speech activity\n$N[k]$"),
              (2.6, 9.6, 3.0, 1.0, "Magnitude\n" r"$|\tilde{X}[k]|$"),
              (7.2, 9.6, 3.0, 1.0, "Magnitude\n$|N[k]|$"),
              (7.2, 7.7, 3.4, 1.0, "Average\n" r"$\mu(k)=E\{|N[k]|\}$"),
              (2.6, 5.9, 4.2, 1.0, "Spectral Subtraction\n"
               r"$\tilde{S}[k]=|\tilde{X}[k]|-\mu(k)$"),
              (2.7, 3.9, 4.6, 1.4, "Half-wave rectification\n"
               r"$\hat{S}[k]=\{\,0$ if $\tilde{S}[k]<0$;  $\tilde{S}[k]$ otherwise$\}$"),
              (6.6, 3.9, 1.5, 1.0, "IFFT\n" r"$\hat{S}[n]$"),
              (8.8, 3.9, 1.5, 1.0, "D-to-A\n" r"$\hat{S}(t)$")):
        _box(ax, *b)
    ax.add_patch(Ellipse((2.6, 7.7), 0.95, 0.85, fc="white", ec="black", lw=1.4))
    ax.text(2.6, 7.7, r"$\Sigma$", ha="center", va="center", fontsize=13)
    ax.text(3.15, 8.15, "+", fontsize=10)
    ax.text(3.15, 7.25, "-", fontsize=12)
    for a in ((2.6, 10.85, 2.6, 10.1), (7.2, 10.85, 7.2, 10.1), (7.2, 9.1, 7.2, 8.2),
              (2.6, 9.1, 2.6, 8.15), (5.5, 7.7, 3.1, 7.7), (2.6, 7.27, 2.6, 6.4),
              (2.6, 5.4, 2.6, 4.6), (5.0, 3.9, 5.85, 3.9), (7.35, 3.9, 8.05, 3.9)):
        _arrow(ax, *a)
    if CAPTIONS:
        ax.text(5.0, 2.6, "Figure 3. Spectral subtraction noise removal flowchart.",
                ha="center", fontsize=9, color=CAPTION)
    save(fig, outdir, "figure3_flowchart.png")


def figure_4(noise, noisy, res, outdir):
    nx, M = np.concatenate([noise, noisy]), len(noise)
    fig = plt.figure(figsize=(8.2, 9.6))
    ax = fig.add_subplot(2, 1, 1)
    ax.plot(nx, lw=0.35, color="black")
    ax.set_xlim(0, len(nx))
    ax.set_ylim(-1.5, 1.5)
    ax.set_xlabel("Discrete-time n")
    _sci_x(ax)
    ax.add_patch(Rectangle((M, -1.05), M, 2.05, fill=False, ls=":", ec="black"))
    ax.add_patch(Rectangle((0, -0.38), M, 0.66, fill=False, ls=":", ec="black"))
    ax.annotate("speech activity", xy=(M + 1500, 0.72), xytext=(0.30 * len(nx), 0.62),
                fontsize=8, arrowprops=dict(arrowstyle="->", lw=0.9))
    ax.annotate("non-speech activity", xy=(0.24 * len(nx), 0.22),
                xytext=(0.20 * len(nx), -0.72), fontsize=8,
                arrowprops=dict(arrowstyle="->", lw=0.9))
    ax.text(-0.09, 1.02, "(a)", transform=ax.transAxes, fontsize=10)
    ax2 = fig.add_subplot(2, 1, 2, projection="3d")
    _mesh(ax2, np.vstack([res["noise_mag"], res["noisy_mag"]]))
    ax2.text2D(-0.02, 0.98, "(b)", transform=ax2.transAxes, fontsize=10)
    _caption(fig, "Figure 4. (a) The signal $NX[n]$ composed of $N[n]$ followed by "
                  "$X[n]$;\n(b) Spectrum $NX[k]$ of $NX[n]$.", y=0.01)
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    save(fig, outdir, "figure4_nx_and_spectrum.png")


def _figure_3d(mag, title, outdir, name):
    fig = plt.figure(figsize=(7.4, 5.6))
    ax = fig.add_subplot(111, projection="3d")
    _mesh(ax, mag)
    if CAPTIONS:
        ax.set_title(title, fontsize=9, color=CAPTION)
    save(fig, outdir, name)


def figure_5(res, outdir):
    _figure_3d(res["s_tilde"], "Figure 5. Spectrum of the noisy speech frames after\n"
               "subtracting the average noise magnitude spectrum.",
               outdir, "figure5_after_subtraction.png")


def figure_6(res, outdir):
    _figure_3d(res["s_hat_mag"], "Figure 6. Spectrum of the speech frames after "
               "subtracting\naverage noise magnitude and half-wave rectification.",
               outdir, "figure6_after_rectification.png")


# Validation Figures

def figure_7_extra(speech, noisy, res, outdir):
    fig, axes = plt.subplots(3, 1, figsize=(7.2, 6.4), sharex=True)
    for ax, sig, lab in zip(axes, (speech, noisy, res["s_hat"]),
                            ("Clean $S[n]$", "Noisy $X[n]$", r"Reconstructed $\hat{S}[n]$")):
        ax.plot(sig, lw=0.4, color="black")
        ax.set_xlim(0, len(sig))
        ax.set_ylim(-1, 1)
        ax.set_ylabel(lab, fontsize=8)
    axes[-1].set_xlabel("Discrete-time n")
    _sci_x(axes[-1])
    fig.suptitle("Validation (not in paper): clean vs. noisy vs. reconstructed", fontsize=10)
    save(fig, outdir, "figure7_validation_waveforms.png")


def _label(r):
    return "%d/%s/%d" % (r["L"], "1-2" if r["overlap"] == "half" else "1-4", r["n_avg"])


def _bar_axes(ax, labels, ylabel, title):
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=8)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("window(pts) / overlap / frames averaged")
    ax.set_title(title)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)


def figure_8_sweep(rows, outdir):
    x = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(x - 0.2, [r["dsnr"] for r in rows], 0.4, label="This reproduction", color=BLUE)
    ax.bar(x + 0.2, [r["paper_dsnr"] for r in rows], 0.4, label="Karam et al. (2014)",
           color=ORANGE)
    _bar_axes(ax, [_label(r) for r in rows], r"$\Delta SNR$ (dB)",
              r"Eq. (12): $\Delta SNR = SNR - SNR_{ref}$ for all 17 configurations")
    save(fig, outdir, "figure8_dsnr_sweep.png")


def figure_9_true_snr(ref, rows, outdir):
    gain = [r["snr_true"] - r["snr_true_in"] for r in [ref] + rows]
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(np.arange(len(gain)), gain, 0.6, color=[GREY] + [BLUE] * len(rows))
    ax.axhline(gain[0], color=GREY, lw=0.8, ls="--", label="reference design")
    _bar_axes(ax, ["256/1-2/1 (ref)"] + [_label(r) for r in rows],
              "Output SNR - input SNR (dB)",
              "Level-invariant check: actual noise-reduction gain of each configuration")
    save(fig, outdir, "figure9_true_snr_gain.png")


# Master Draw Function

def draw_all(speech, noise, noisy, ref, rows, outdir, noise_label):
    figure_1(outdir)
    figure_2(speech, noise, noisy, outdir, noise_label)
    figure_3(outdir)
    figure_4(noise, noisy, ref, outdir)
    figure_5(ref, outdir)
    figure_6(ref, outdir)
    figure_7_extra(speech, noisy, ref, outdir)
    figure_8_sweep(rows, outdir)
    figure_9_true_snr(ref, rows, outdir)
