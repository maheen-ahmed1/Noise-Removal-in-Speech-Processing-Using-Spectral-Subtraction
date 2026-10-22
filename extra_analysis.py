import argparse
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import spectral_subtraction as ss
import figures


# Helpers

def ref_and_best(speech, noise, noise_est=None):
    noisy = speech + noise
    return (ss.run_configuration(speech, noise, noisy, 256, "half", 1, noise_est=noise_est),
            ss.run_configuration(speech, noise, noisy, 256, "quarter", 1, noise_est=noise_est))


def gain(r):
    return r["snr_true"] - r["snr_true_in"]


def slide(n_total, n, step_ms=10):
    return np.arange(0, n_total - n + 1, int(step_ms * 1e-3 * ss.FS))


# Experiments

def experiment_a(speech, noise_raw, n, outdir):
    offsets = slide(len(noise_raw), n)
    trials = [ref_and_best(speech, ss.fit_length(noise_raw[o:], n, 0.30)) for o in offsets]
    ref_snr = np.array([ref["snr"] for ref, win in trials])
    dsnr = np.array([win["snr"] - ref["snr"] for ref, win in trials])
    g_ref = np.array([gain(ref) for ref, win in trials])
    g_win = np.array([gain(win) for ref, win in trials])

    t = offsets / ss.FS * 1000
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(t, dsnr, "o-", color=figures.BLUE, ms=3,
               label="this reproduction (each noise slot)")
    ax[0].axhline(0.3371, color=figures.ORANGE, ls="--", label="paper's headline +0.3371 dB")
    ax[0].axhline(0, color="black", lw=0.8)
    ax[0].set_ylabel(r"Eq. (12) $\Delta SNR$, 256/one-fourth/none (dB)")
    ax[0].set_title("Repeatability of the headline comparison")
    ax[1].plot(t, g_ref, "o-", ms=3, color=figures.GREY, label="reference 256/one-half/none")
    ax[1].plot(t, g_win, "s-", ms=3, color=figures.BLUE, label="256/one-fourth/none")
    ax[1].set_ylabel("output SNR - input SNR (dB)")
    ax[1].set_title("Actual noise reduction, same trials")
    for a in ax:
        a.set_xlabel("noise segment start (ms)")
        a.legend(fontsize=8)
        a.grid(alpha=0.3)
    figures.save(fig, outdir, "figureA_repeatability.png")

    sd = lambda v: v.std(ddof=1)
    return ["Repeatability: %d trials, noise window slid in 10 ms steps over %.0f ms "
            "of noise" % (len(offsets), len(noise_raw) / ss.FS * 1000),
            "   SNR_ref (Eq. 9)                  : %.4f +/- %.4f dB (mean +/- 1 s.d.)"
            % (ref_snr.mean(), sd(ref_snr)),
            "   dSNR 256/one-fourth/none vs ref  : %+.4f +/- %.4f dB  (range %+.4f to %+.4f)"
            % (dsnr.mean(), sd(dsnr), dsnr.min(), dsnr.max()),
            "   paper's single-trial value       : +0.3371 dB",
            "   actual gain, reference           : %+.3f +/- %.3f dB" % (g_ref.mean(), sd(g_ref)),
            "   actual gain, 256/one-fourth/none : %+.3f +/- %.3f dB" % (g_win.mean(), sd(g_win))]


def experiment_a2(s_raw, noise, n):
    offsets = slide(len(s_raw), n)
    d = np.array([win["snr"] - ref["snr"] for ref, win in
                  (ref_and_best(ss.fit_length(s_raw[o:], n, 0.92), noise) for o in offsets)])
    return ["A2. Speech window slid in 10 ms steps (%d trials, different speech content "
            "each time)" % len(offsets),
            "   dSNR 256/one-fourth/none vs ref  : %+.4f +/- %.4f dB  (range %+.4f to %+.4f)"
            % (d.mean(), d.std(ddof=1), d.min(), d.max())]


def experiment_b(speech, noise_raw, n):
    added = ss.fit_length(noise_raw, n)
    scale = 0.30 / np.max(np.abs(added))
    added, other = added * scale, noise_raw[n:] * scale
    o_ref, o_win = ref_and_best(speech, added)
    i_ref, i_win = ref_and_best(speech, added, noise_est=other)
    return ["B. Noise estimate source (independent estimate uses the last %.0f ms of "
            "the recording)" % (len(other) / ss.FS * 1000),
            "   actual gain, reference  : oracle %+.3f dB   independent %+.3f dB"
            % (gain(o_ref), gain(i_ref)),
            "   actual gain, 256/1-4/1  : oracle %+.3f dB   independent %+.3f dB"
            % (gain(o_win), gain(i_win)),
            "   Eq.(9) SNR_ref          : oracle %.4f dB  independent %.4f dB"
            % (o_ref["snr"], i_ref["snr"])]


def experiment_c(speech, noise_raw, n, outdir):
    noise_u = ss.fit_length(noise_raw, n)
    targets = np.arange(-5, 21, 2.5)
    runs = []
    for snr_in in targets:
        g = np.sqrt(np.mean(speech ** 2) / (np.mean(noise_u ** 2) * 10 ** (snr_in / 10)))
        runs.append(ss.run_configuration(speech, noise_u * g, speech + noise_u * g))
    eq9 = np.array([r["snr"] for r in runs])

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(targets, [gain(r) for r in runs], "o-", color=figures.BLUE,
            label="actual improvement (gain-matched output - input SNR)")
    ax.plot(targets, eq9, "s--", color=figures.ORANGE, label="paper's Eq. (9) SNR of the output")
    ax.plot(targets, targets, ":", color=figures.GREY,
            label="input SNR = 10 log(P_s / P_n)  (identity line)")
    ax.set_xlabel("input SNR = 10 log(P_s / P_n) (dB), set by scaling the noise")
    ax.set_ylabel("dB")
    ax.set_title("Reference design across noise levels")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    figures.save(fig, outdir, "figureC_input_snr_sweep.png")

    lines = ["C. Input-SNR sweep (reference design). 'set' = 10log(Ps/Pn); "
             "in/out = gain-matched SNR",
             "   %8s%9s%9s%13s%12s" % ("set SNR", "in SNR", "out SNR", "improvement",
                                        "Eq.(9) SNR")]
    for t, r, e in zip(targets, runs, eq9):
        lines.append("   %8.1f%9.2f%9.2f%+13.2f%12.2f"
                     % (t, r["snr_true_in"], r["snr_true"], gain(r), e))
    return lines


def experiment_d(speech, noise, outdir):
    ref = ss.run_configuration(speech, noise, speech + noise)
    fig, axes = plt.subplots(3, 1, figsize=(8, 7.5), sharex=True)
    for ax, sig, lab in zip(axes, (speech, speech + noise, ref["s_hat"]),
                            ("(a) clean speech S[n]", "(b) noisy speech X[n]",
                             "(c) after spectral subtraction")):
        ax.specgram(sig, NFFT=512, Fs=ss.FS, noverlap=384, cmap="magma", vmin=-120, vmax=-20)
        ax.set_ylim(0, 8000)
        ax.set_ylabel("Hz")
        ax.set_title(lab, fontsize=9, loc="left")
    axes[-1].set_xlabel("time (s)")
    fig.suptitle("Spectrograms (reference design). Isolated specks in (c) are residual "
                 "'musical noise'", fontsize=10)
    figures.save(fig, outdir, "figureD_spectrograms.png")


# Main

def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--speech", default=os.path.join(HERE, "speech.wav"))
    p.add_argument("--noise", default=os.path.join(HERE, "noise.wav"))
    p.add_argument("--outdir", default=os.path.join(HERE, "analysis"))
    p.add_argument("--no-show", action="store_true")
    args = p.parse_args(argv)
    os.makedirs(args.outdir, exist_ok=True)
    figures.setup(show=not args.no_show)

    n = ss.N_SAMPLES
    s_raw, _ = ss.read_active(args.speech)
    noise_raw, _ = ss.read_active(args.noise)
    speech = ss.fit_length(s_raw, n, 0.92)
    noise = ss.fit_length(noise_raw, n, 0.30)

    lines = ["EXTRA ANALYSES (not in the paper)", ""]
    lines += experiment_a(speech, noise_raw, n, args.outdir)
    lines += experiment_a2(s_raw, noise, n) + [""]
    lines += experiment_b(speech, noise_raw, n) + [""]
    experiment_d(speech, noise, args.outdir)
    lines += experiment_c(speech, noise_raw, n, args.outdir)

    text = "\n".join(lines)
    print(text)
    with open(os.path.join(args.outdir, "extra_analysis.txt"), "w", encoding="utf-8") as f:
        f.write(text + "\n")
    figures.show()
    return 0


if __name__ == "__main__":
    sys.exit(main())
