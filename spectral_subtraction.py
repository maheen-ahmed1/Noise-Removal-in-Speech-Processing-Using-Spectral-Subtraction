import argparse
import csv
import os
import sys

import numpy as np
from scipy.io import wavfile
from scipy.signal import lfilter, resample_poly

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import figures

# Constants
FS = 44100
N_SAMPLES = 28446
EPS = 1e-12

PAPER_SNR = {
    (256, "half", 1): 13.0248,
    (256, "half", 3): 12.4914, (256, "half", 6): 12.4538,
    (256, "quarter", 1): 13.3619, (256, "quarter", 3): 12.3457,
    (256, "quarter", 6): 12.6906,
    (128, "half", 1): 12.9805, (128, "half", 3): 12.2872,
    (128, "half", 6): 12.2080, (128, "quarter", 1): 13.2412,
    (128, "quarter", 3): 10.5704, (128, "quarter", 6): 12.2584,
    (512, "half", 1): 13.1032, (512, "half", 3): 13.0085,
    (512, "half", 6): 12.9394, (512, "quarter", 1): 5.1722,
    (512, "quarter", 3): 5.8646, (512, "quarter", 6): 6.5488,
}
REF = (256, "half", 1)


# Audio I/O

def read_active(path, fs=FS, start=None):
    fs_in, data = wavfile.read(path)
    x = data.astype(np.float64)
    if np.issubdtype(data.dtype, np.integer):
        x /= np.iinfo(data.dtype).max
    if x.ndim > 1:
        x = x.mean(axis=1)
    if fs_in != fs:
        g = np.gcd(int(fs_in), int(fs))
        x = resample_poly(x, fs // g, fs_in // g)
    loud = np.flatnonzero(np.abs(x) > 1e-3)
    if loud.size == 0:
        raise ValueError(path + " contains no sound above -60 dBFS")
    a = loud[0] if start is None else int(round(start * fs))
    return x[a:loud[-1] + 1], int(a)


def fit_length(x, n_samples, peak=None):
    x = np.concatenate([x[:n_samples], np.zeros(max(0, n_samples - len(x)))])
    if peak is not None and np.max(np.abs(x)) > EPS:
        x = x * peak / np.max(np.abs(x))
    return x


def write_wav(path, x):
    wavfile.write(path, FS, (np.clip(x, -1, 1) * 32767).astype(np.int16))


def make_demo_audio():
    rng = np.random.default_rng(0)
    n = np.arange(N_SAMPLES)
    t = n / FS
    phase = 2 * np.pi * np.cumsum(118 * (1 + 0.02 * np.sin(2 * np.pi * 3.1 * t))) / FS
    src = sum(np.sin(h * phase + rng.uniform(0, 2 * np.pi)) / h ** 1.45
              for h in range(1, 19))
    src /= np.max(np.abs(src))

    def resonator(x, fc, bw):
        r = np.exp(-np.pi * bw / FS)
        return lfilter([1 - r], [1, -2 * r * np.cos(2 * np.pi * fc / FS), r * r], x)

    voiced = resonator(src, 690, 90) + 0.55 * resonator(src, 1180, 110) \
        + 0.30 * resonator(src, 2580, 150)
    voiced /= np.max(np.abs(voiced))
    fric = resonator(rng.normal(0, 1, N_SAMPLES), 4200, 2200)
    fric /= np.max(np.abs(fric))

    def bump(c, w, p=2.0):
        return np.exp(-0.5 * np.abs((n - c) / w) ** p)

    speech = (0.58 * bump(7600, 2500, 2.4) + 1.0 * bump(20200, 2900, 2.4)
              + 0.12 * bump(13200, 700)) * voiced \
        + (0.26 * bump(4100, 750) + 0.24 * bump(16300, 650)
           + 0.20 * bump(24800, 900)) * fric

    rng = np.random.default_rng(1)
    shaped = lfilter([0.35, 0.30, 0.20, 0.10], [1.0, -0.55],
                     rng.normal(0, 1, N_SAMPLES))
    noise = 0.85 * shaped / np.max(np.abs(shaped)) + 0.18 * np.sin(2 * np.pi * 108 * t) \
        + 0.12 * np.sin(2 * np.pi * 216 * t) + 0.07 * np.sin(2 * np.pi * 432 * t)
    return fit_length(speech, N_SAMPLES, 0.92), fit_length(noise, N_SAMPLES, 0.30)


# Framing, Windowing, FFT

def hanning_window(L):
    n = np.arange(L)
    return 0.5 - 0.5 * np.cos(2 * np.pi * n / L)


def hop_size(L, overlap, dense_quarter=False):
    if overlap == "half":
        return L // 2
    return L // 4 if dense_quarter else 3 * L // 4


def analyze(x, L, hop):
    n_frames = 1 + (len(x) - L) // hop
    idx = np.arange(L)[None, :] + hop * np.arange(n_frames)[:, None]
    frames = x[idx] * hanning_window(L)
    return np.fft.rfft(frames, n=L, axis=1)


# Spectral Subtraction

def average_frames(mag, n_avg, mode="centered"):
    if n_avg <= 1:
        return mag
    out = np.empty_like(mag)
    for i in range(len(mag)):
        start = i - n_avg + 1 if mode == "causal" else i - n_avg // 2
        out[i] = mag[max(start, 0):start + n_avg].mean(axis=0)
    return out


def synthesize(mag, phase, L, hop, out_len, synthesis_window=False,
               ola_normalise=True):
    frames = np.fft.irfft(mag * np.exp(1j * phase), n=L, axis=1)
    w = hanning_window(L)
    if synthesis_window:
        frames, w = frames * w, w ** 2
    total = (len(frames) - 1) * hop + L
    y, wsum = np.zeros(total), np.zeros(total)
    for i, frame in enumerate(frames):
        y[i * hop:i * hop + L] += frame
        wsum[i * hop:i * hop + L] += w
    if ola_normalise:
        cola = np.median(wsum[L:total - L]) if total > 2 * L else np.median(wsum)
        y = y / np.maximum(wsum, 0.5 * cola)
    return fit_length(y, out_len)


def run_configuration(speech, noise, noisy, L=256, overlap="half", n_avg=1,
                      synthesis_window=False, collect_spectra=False,
                      avg_mode="centered", ola_normalise=True, noise_est=None,
                      dense_quarter=False):
    hop = hop_size(L, overlap, dense_quarter)
    noise_spec = analyze(noise if noise_est is None else noise_est, L, hop)
    noisy_spec = analyze(noisy, L, hop)

    mu = np.mean(np.abs(noise_spec), axis=0)
    mag = average_frames(np.abs(noisy_spec), n_avg, avg_mode)
    s_tilde = mag - mu[None, :]
    s_hat_mag = np.maximum(s_tilde, 0.0)
    s_hat = synthesize(s_hat_mag, np.angle(noisy_spec), L, hop,
                       len(noisy), synthesis_window, ola_normalise)

    res = {"L": L, "overlap": overlap, "n_avg": n_avg, "hop": hop,
           "n_frames": len(noisy_spec), "s_hat": s_hat,
           "snr": snr_db(s_hat, noise),
           "snr_true": snr_true_db(speech, s_hat),
           "snr_true_in": snr_true_db(speech, noisy)}
    if collect_spectra:
        res.update(noise_mag=np.abs(noise_spec), noisy_mag=np.abs(noisy_spec),
                   mu=mu, s_tilde=s_tilde, s_hat_mag=s_hat_mag)
    return res


# Scoring

def rms(x):
    return float(np.sqrt(np.mean(np.square(x))))


def snr_db(s_hat, noise):
    return 20 * np.log10(max(rms(s_hat), EPS) / max(rms(noise), EPS))


def snr_true_db(clean, s_hat):
    g = np.dot(clean, s_hat) / max(np.dot(s_hat, s_hat), EPS)
    err = clean - g * s_hat
    return 10 * np.log10(max(np.dot(clean, clean), EPS) / max(np.dot(err, err), EPS))


def run_sweep(speech, noise, noisy, **opts):
    ref = run_configuration(speech, noise, noisy, *REF, **opts)
    rows = []
    for table, L in ((1, 256), (2, 128), (3, 512)):
        for overlap in ("half", "quarter"):
            for n_avg in (1, 3, 6):
                if (L, overlap, n_avg) == REF:
                    continue
                r = run_configuration(speech, noise, noisy, L, overlap, n_avg, **opts)
                r["table"] = table
                r["dsnr"] = r["snr"] - ref["snr"]
                r["paper_snr"] = PAPER_SNR[(L, overlap, n_avg)]
                r["paper_dsnr"] = r["paper_snr"] - PAPER_SNR[REF]
                rows.append(r)
    return ref, ref["snr"], rows


# Report

def format_tables(snr_ref, rows):
    ov_name = {"half": "One-half", "quarter": "One-fourth"}
    avg_name = {1: "No frame averaging", 3: "3 frames averaging",
                6: "6 frames averaging"}
    out = ["Reference design (Sec. 4.3): 256-pt Hanning, one-half overlapped, "
           "no frame averaging",
           "    SNR_ref  (this reproduction) = %8.4f dB" % snr_ref,
           "    SNR_ref  (paper)             = %8.4f dB" % PAPER_SNR[REF], ""]
    for table, L in ((1, 256), (2, 128), (3, 512)):
        out += ["Table %d. Noise removal design using %d points Hanning window."
                % (table, L),
                "%-44s%10s%11s%12s%12s" % ("Configuration", "SNR(dB)",
                                           "dSNR(dB)", "paper SNR", "paper dSNR")]
        for r in rows:
            if r["table"] == table:
                cfg = "%dpt / %s overlap / %s" % (L, ov_name[r["overlap"]],
                                                  avg_name[r["n_avg"]])
                out.append("%-44s%10.4f%+11.4f%12.4f%+12.4f" % (
                    cfg, r["snr"], r["dsnr"], r["paper_snr"], r["paper_dsnr"]))
        out += [""]
    return "\n".join(out)


def write_report(args, src, offsets, speech, noise, ref, rows, outdir):
    doubled = 2.0 * ref["s_hat"]
    level = ("none (raw file levels)" if args.no_normalise else
             "speech peak %s, noise peak %s" % (args.speech_peak, args.noise_peak))
    L = ["REPRODUCTION OF Karam et al. (2014), JSIP 5, 32-41",
         "Noise Removal in Speech Processing Using Spectral Subtraction",
         "",
         "Audio source        : " + src]
    if offsets:
        for name, off, avail in (("Speech used from    ", offsets[0], offsets[1]),
                                 ("Noise used from     ", offsets[2], offsets[3])):
            L.append("%s: sample %d (%.3f s into file); %.1f ms of sound available"
                     % (name, off, off / FS, avail / FS * 1000))
    L += ["Sampling rate       : %d Hz" % FS,
          "Samples per signal  : %d  (%.1f ms)" % (len(speech), 1000 * len(speech) / FS),
          "Level normalisation : " + level,
          "Options             : quarter-reading=%s, frame-avg-mode=%s, "
          "synthesis-window=%s, ola-normalise=%s" % (
              args.quarter_reading, args.frame_avg_mode, args.synthesis_window,
              not args.no_ola_norm),
          "Reference framing   : L=256, hop=128, %d frames  "
          "(paper reports 221 for 28,446 samples)" % ref["n_frames"], "",
          "Reference-configuration results",
          "  Eq.(9)  SNR  (paper's definition)      : %8.4f dB" % ref["snr"],
          "  Conventional output SNR                : %8.4f dB" % ref["snr_true"],
          "  Conventional input  SNR (unprocessed)  : %8.4f dB" % ref["snr_true_in"],
          "  Actual improvement                     : %+8.4f dB"
          % (ref["snr_true"] - ref["snr_true_in"]), "",
          "Metric sanity check: multiply the denoised output by 2 (removes no noise)",
          "  change in Eq.(9) SNR                   : %+8.4f dB"
          % (snr_db(doubled, noise) - ref["snr"]),
          "  change in conventional SNR             : %+8.4f dB"
          % (snr_true_db(speech, doubled) - ref["snr_true"]), "",
          format_tables(ref["snr"], rows)]
    best = max(rows, key=lambda r: r["dsnr"])
    best_t = max(rows, key=lambda r: r["snr_true"])
    L += ["Best configuration in THIS run : %d-pt window, %s overlap, %d frame(s) "
          "averaged  (dSNR = %+.4f dB)" % (best["L"], best["overlap"],
                                           best["n_avg"], best["dsnr"]),
          "Best by conventional SNR       : %d-pt window, %s overlap, %d frame(s) "
          "averaged  (output SNR = %.4f dB vs ref %.4f dB)" % (
              best_t["L"], best_t["overlap"], best_t["n_avg"],
              best_t["snr_true"], ref["snr_true"]),
          "Best configuration in the PAPER: 256-pt window, one-fourth overlap, "
          "no frame averaging (dSNR = +0.3371 dB)",
          "NOTE: the paper's ABSTRACT says 128 points here; its CONCLUSION and "
          "Table 1 say 256. Table 1 is self-consistent, so 256 is correct and "
          "the abstract contains an error."]
    text = "\n".join(L)
    print(text)
    with open(os.path.join(outdir, "results_report.txt"), "w", encoding="utf-8") as f:
        f.write(text + "\n")

    with open(os.path.join(outdir, "results_table.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["table", "window_points", "overlap", "frames_averaged", "hop",
                    "n_frames", "snr_db", "dsnr_db", "paper_snr_db",
                    "paper_dsnr_db", "true_output_snr_db", "true_input_snr_db"])
        for r in [dict(ref, table=0, dsnr=0.0, paper_snr=PAPER_SNR[REF],
                       paper_dsnr=0.0)] + rows:
            w.writerow([r["table"], r["L"], r["overlap"], r["n_avg"], r["hop"],
                        r["n_frames"], "%.4f" % r["snr"], "%.4f" % r["dsnr"],
                        r["paper_snr"], "%.4f" % r["paper_dsnr"],
                        "%.4f" % r["snr_true"], "%.4f" % r["snr_true_in"]])


# Main

def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--speech")
    p.add_argument("--noise")
    p.add_argument("--outdir", default=os.path.join(HERE, "results"))
    p.add_argument("--demo", action="store_true")
    p.add_argument("--speech-peak", type=float, default=0.92)
    p.add_argument("--noise-peak", type=float, default=0.30)
    p.add_argument("--no-normalise", action="store_true")
    p.add_argument("--speech-start", type=float)
    p.add_argument("--noise-start", type=float)
    p.add_argument("--full-length", action="store_true")
    p.add_argument("--frame-avg-mode", choices=["centered", "causal"], default="centered")
    p.add_argument("--quarter-reading", choices=["sparse", "dense"], default="sparse")
    p.add_argument("--synthesis-window", action="store_true")
    p.add_argument("--no-ola-norm", action="store_true")
    p.add_argument("--noise-label")
    p.add_argument("--no-show", action="store_true")
    args = p.parse_args(argv)
    os.makedirs(args.outdir, exist_ok=True)
    figures.setup(show=not args.no_show)

    if not (args.speech or args.noise or args.demo):
        s_def, n_def = os.path.join(HERE, "speech.wav"), os.path.join(HERE, "noise.wav")
        if os.path.isfile(s_def) and os.path.isfile(n_def):
            args.speech, args.noise = s_def, n_def
            print("No --speech/--noise given: using %s and %s" % (
                os.path.basename(s_def), os.path.basename(n_def)))
    offsets = None
    if args.speech and args.noise:
        s_raw, s_off = read_active(args.speech, start=args.speech_start)
        n_raw, n_off = read_active(args.noise, start=args.noise_start)
        n = min(len(s_raw), len(n_raw)) if args.full_length else N_SAMPLES
        speech = fit_length(s_raw, n, None if args.no_normalise else args.speech_peak)
        noise = fit_length(n_raw, n, None if args.no_normalise else args.noise_peak)
        offsets = (s_off, len(s_raw), n_off, len(n_raw))
        src = "%s + %s" % (os.path.basename(args.speech), os.path.basename(args.noise))
    else:
        print("!! Using SYNTHETIC demo audio (no speech.wav/noise.wav found, "
              "or --demo given).\n!! Not valid for submission.")
        speech, noise = make_demo_audio()
        src = "synthetic demo audio"
    noisy = speech + noise

    opts = dict(synthesis_window=args.synthesis_window, avg_mode=args.frame_avg_mode,
                ola_normalise=not args.no_ola_norm,
                dense_quarter=args.quarter_reading == "dense")
    ref = run_configuration(speech, noise, noisy, *REF, collect_spectra=True, **opts)
    _, _, rows = run_sweep(speech, noise, noisy, **opts)

    for name, sig in (("01_clean_speech", speech), ("02_noise", noise),
                      ("03_noisy_speech", noisy), ("04_denoised_speech", ref["s_hat"])):
        write_wav(os.path.join(args.outdir, name + ".wav"), sig)

    write_report(args, src, offsets, speech, noise, ref, rows, args.outdir)
    label = args.noise_label or ("Hair-dryer noise" if offsets else
                                 "Vacuum cleaner noise")
    figures.draw_all(speech, noise, noisy, ref, rows, args.outdir, label)
    print("\nAll figures, WAVs, report and CSV written to: "
          + os.path.abspath(args.outdir))
    figures.show()
    return 0


if __name__ == "__main__":
    sys.exit(main())
