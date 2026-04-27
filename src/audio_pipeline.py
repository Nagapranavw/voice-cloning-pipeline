"""
audio_pipeline.py

Processes a folder of local WAV files for use as voice cloning reference data.

This pipeline assumes you already have WAV recordings of the speaker you're
cloning — for example, recordings of yourself, or files you have explicit
consent to use. It does not include any functionality for downloading audio
from third-party platforms.

Steps for each WAV:
  1. Separate the voice track from background (music, noise) using Demucs
  2. Remove silence gaps longer than 1.5 seconds
  3. Detect amplitude spikes (claps, crowd noise) that would corrupt training data
  4. Save cleaned output and write a timestamped review report

Usage:
    python src/audio_pipeline.py --input recordings/ --output output/
"""

import os
import re
import sys
import shutil
import subprocess
import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
from scipy.io import wavfile
from pydub import AudioSegment
from pydub.silence import split_on_silence   # strip_silence does not exist in pydub


# ── Config ────────────────────────────────────────────────────────────────────

SILENCE_THRESH_DBFS  = -40    # silence below this level gets trimmed
MIN_SILENCE_LEN_MS   = 1500   # gaps longer than 1.5s are removed
SPIKE_THRESHOLD      = 0.95   # amplitude ratio to flag a spike (0–1)
SPIKE_MIN_DISTANCE_S = 3.0    # minimum seconds between flagged spikes


# ── Helpers ───────────────────────────────────────────────────────────────────

def run(cmd: str) -> bool:
    """Run a shell command. Returns True on success, False on failure."""
    result = subprocess.run(cmd, shell=True)
    return result.returncode == 0


def sanitize_filename(name: str) -> str:
    """Remove characters that cause issues on Windows / Linux filesystems."""
    return re.sub(r'[\\/*?:"<>|]', "_", name)


def find_wavs(input_dir: str) -> list[Path]:
    """Return all .wav files in input_dir (non-recursive)."""
    wavs = sorted(Path(input_dir).glob("*.wav"))
    if not wavs:
        print(f"No .wav files found in: {input_dir}")
    return wavs


# ── Voice separation ──────────────────────────────────────────────────────────

def separate_voice(wav_path: str, output_dir: str) -> str:
    """
    Use Demucs (htdemucs model) to isolate the vocals track from a WAV file.
    Returns path to the separated vocals .wav.

    Falls back to the original file if Demucs fails.
    Demucs downloads ~80MB of model weights on first run.

    Demucs writes output relative to the --out flag we pass, so we pass
    output_dir explicitly and construct the expected output path from there.
    Demucs also dislikes special characters in filenames, so the input is
    copied to a safe temp name (demucs_input.wav) before processing.
    """
    safe_name = "demucs_input.wav"
    safe_path = os.path.join(output_dir, safe_name)
    shutil.copy2(wav_path, safe_path)

    # Pass --out so Demucs writes into output_dir, not the cwd
    cmd = (
        f'python -m demucs --two-stems=vocals -n htdemucs '
        f'--out "{output_dir}" "{safe_path}"'
    )
    print(f"  → Separating voice: {os.path.basename(wav_path)}")

    if not run(cmd):
        print(f"  ! Demucs failed — using original audio")
        os.remove(safe_path)
        return wav_path

    # Demucs writes to: <out>/htdemucs/<stem>/vocals.wav
    # stem = filename without extension = "demucs_input"
    vocals_path = os.path.join(output_dir, "htdemucs", "demucs_input", "vocals.wav")
    os.remove(safe_path)

    if os.path.exists(vocals_path):
        return vocals_path

    print(f"  ! Demucs output not found — using original audio")
    return wav_path


# ── Silence removal ───────────────────────────────────────────────────────────

def remove_silence(wav_path: str, output_path: str) -> str:
    """
    Strip silence gaps longer than MIN_SILENCE_LEN_MS using pydub's
    split_on_silence, then reassemble chunks with 200ms padding.
    Saves result to output_path and returns it.
    """
    print(f"  → Removing silence: {os.path.basename(wav_path)}")
    audio = AudioSegment.from_wav(wav_path)

    chunks = split_on_silence(
        audio,
        silence_thresh=SILENCE_THRESH_DBFS,
        min_silence_len=MIN_SILENCE_LEN_MS,
        keep_silence=200,   # 200ms padding around each chunk
    )

    if not chunks:
        # Nothing was split — export as-is
        combined = audio
    else:
        combined = chunks[0]
        for chunk in chunks[1:]:
            combined += chunk

    combined.export(output_path, format="wav")
    return output_path


# ── Spike detection ───────────────────────────────────────────────────────────

def detect_spikes(wav_path: str) -> list[float]:
    """
    Return timestamps (in seconds) where amplitude spikes occur.
    Spikes typically indicate clapping, crowd noise, or audio artifacts
    that would corrupt voice cloning reference data.
    """
    sample_rate, data = wavfile.read(wav_path)

    if data.ndim > 1:
        data = data.mean(axis=1)  # mix to mono

    data = data.astype(np.float32)
    peak = np.abs(data).max()
    if peak == 0:
        return []

    normalized = np.abs(data) / peak
    spike_min_samples = int(SPIKE_MIN_DISTANCE_S * sample_rate)

    spikes = []
    last_spike = -spike_min_samples

    for i, val in enumerate(normalized):
        if val >= SPIKE_THRESHOLD and (i - last_spike) >= spike_min_samples:
            spikes.append(round(i / sample_rate, 2))
            last_spike = i

    return spikes


# ── Per-file pipeline ─────────────────────────────────────────────────────────

def process_wav(wav_path: Path, output_dir: str) -> dict:
    """
    Run the full pipeline on a single WAV file:
      separate voice → remove silence → detect spikes

    Returns a result dict used to build the review report.
    """
    result = {"file": str(wav_path), "status": "ok", "spikes": [], "output": None}

    work_dir  = os.path.join(output_dir, "work")
    clean_dir = os.path.join(output_dir, "cleaned")
    os.makedirs(work_dir, exist_ok=True)
    os.makedirs(clean_dir, exist_ok=True)

    stem = sanitize_filename(wav_path.stem)

    # 1. Separate voice (Demucs writes into work_dir)
    vocals_wav = separate_voice(str(wav_path), work_dir)

    # 2. Remove silence
    clean_wav = os.path.join(clean_dir, f"{stem}_clean.wav")
    remove_silence(vocals_wav, clean_wav)

    # 3. Detect spikes
    spikes = detect_spikes(clean_wav)
    if spikes:
        print(f"  ⚠ Spikes at: {spikes[:5]}{'...' if len(spikes) > 5 else ''} seconds")

    result["output"] = clean_wav
    result["spikes"] = spikes
    if len(spikes) > 10:
        result["status"] = "noisy"

    return result


# ── Report ────────────────────────────────────────────────────────────────────

def write_report(results: list[dict], output_dir: str):
    """Write a timestamped review report summarising pipeline output."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = os.path.join(output_dir, f"review_report_{ts}.txt")

    with open(report_path, "w") as f:
        f.write(f"Audio Pipeline Report — {ts}\n")
        f.write("=" * 60 + "\n\n")
        for r in results:
            f.write(f"File   : {r['file']}\n")
            f.write(f"Status : {r['status']}\n")
            f.write(f"Output : {r.get('output', 'N/A')}\n")
            if r["spikes"]:
                f.write(f"Spikes : {r['spikes']}\n")
            f.write("\n")

    print(f"\n✓ Report saved: {report_path}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Preprocess local WAV files for voice cloning reference data"
    )
    parser.add_argument(
        "--input", required=True,
        help="Folder containing input .wav files (your own recordings)"
    )
    parser.add_argument(
        "--output", default="output",
        help="Output directory for cleaned files and report (default: output/)"
    )
    args = parser.parse_args()

    wavs = find_wavs(args.input)
    if not wavs:
        sys.exit(1)

    print(f"\nFound {len(wavs)} WAV file(s) in '{args.input}'.\n")

    results = []
    for i, wav in enumerate(wavs, 1):
        ts = datetime.now().strftime("%H:%M:%S")
        print(f"\n══ [{ts}] PROCESSING {i}/{len(wavs)}: {wav.name}")
        result = process_wav(wav, args.output)
        results.append(result)

    write_report(results, args.output)
    ok = sum(1 for r in results if r["status"] == "ok")
    print(f"\nDone. {ok}/{len(results)} files processed cleanly.")
    print(f"Cleaned files → {os.path.join(args.output, 'cleaned')}/")


if __name__ == "__main__":
    main()
