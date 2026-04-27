# voice-cloning-pipeline

An end-to-end pipeline for cleaning voice recordings and synthesizing speech in a cloned voice using Fish Audio S2 Pro — running fully locally on GPU.

The examples in this repo use my own voice as the reference speaker. This pipeline assumes you already have WAV files for the speaker you're cloning. It does not include code or instructions for scraping audio from third-party platforms — you should only ever feed it recordings you made yourself or have explicit consent to use.

---

## What This Does

Given a folder of WAV files (your own recordings, or other consented audio), this pipeline:

1. **Separates the voice track** — Strips background music and noise using Demucs, so the model gets clean speech only
2. **Removes silence** — Trims gaps longer than 1.5 seconds, keeping 200ms of padding around speech so nothing gets clipped
3. **Detects noise spikes** — Flags timestamps where clapping, crowd noise, or audio artifacts would corrupt training data
4. **Synthesizes speech** — Runs Fish Audio S2 Pro as a local API server; takes a cleaned reference clip and generates new audio in that voice from any text input
5. **Writes a report** — Timestamped review of quality flags per file

Designed to run on Ubuntu with a CUDA GPU. Windows WSL2 is also supported (GPU 0 only under WSL).

---

## Why I Built This

I wanted to understand the full stack of a voice cloning pipeline beyond just calling a hosted API — what makes reference audio good or bad, how source separation actually affects model output, and where the real bottlenecks are. Building this end-to-end was a much better learning experience than any tutorial.

---

## Features

- Local WAV folder as input — no external services or platform scraping
- Voice/background separation via Demucs (`htdemucs` model)
- Silence removal with configurable threshold and padding
- Spike/noise detection with configurable threshold and minimum distance between flags
- Fish Audio S2 Pro local inference (via `tools/api_server.py` from the fish-speech repo)
- Zero-shot voice cloning with reference audio injection
- Per-file review report with quality flags
- Works on Ubuntu (Python 3.11 conda env) and Windows WSL2

---

## Tech Stack

| Component | Tool |
|---|---|
| Language | Python 3.11 |
| Voice separation | `demucs` (htdemucs model) |
| Audio processing | `pydub`, `numpy`, `scipy` |
| Voice model | Fish Audio S2 Pro (local) |
| Inference server | `tools/api_server.py` from fish-speech repo |
| HTTP client | `httpx` |
| GPU | CUDA (tested on RTX 3080 / Vast.ai) |

---

## Setup

### 1. Clone the repo

```bash
git clone git@github.com:yourusername/voice-cloning-pipeline.git
cd voice-cloning-pipeline
```

### 2. Create environment (Python 3.11 required — pydub is broken on 3.13)

```bash
conda create -n audio311 python=3.11 -y
conda activate audio311
pip install -r requirements.txt
sudo apt install ffmpeg -y
```

### 3. Download Fish Audio S2 Pro model weights

```bash
git clone https://github.com/fishaudio/fish-speech.git
cd fish-speech
hf download fishaudio/s2-pro --local-dir checkpoints/s2-pro
```

> This downloads ~11GB of model weights (two safetensor shards + codec.pth).

### Optional: one-shot setup & test script

If you're on Ubuntu or Vast.ai and already have Conda installed, you can use the helper script instead of the three steps above:

```bash
chmod +x scripts/setup_and_test.sh
./scripts/setup_and_test.sh
```

This will install all dependencies, download the model, start the server, and run a synthetic end-to-end test automatically.

---

## Usage

### Step 1 — Prepare your recordings

Put your WAV files into a folder (e.g. `recordings/`). These should be recordings of your own voice, or audio you have explicit consent to use. Aim for at least 10–30 seconds of clean, single-speaker audio per file.

### Step 2 — Run the audio pipeline

```bash
python src/audio_pipeline.py --input recordings/ --output output/
```

Cleaned `.wav` files appear in `output/cleaned/` and a `review_report.txt` is saved to `output/`.

### Step 3 — Start the S2 Pro inference server

```bash
cd fish-speech
python tools/api_server.py \
  --llama-checkpoint-path checkpoints/s2-pro \
  --decoder-checkpoint-path checkpoints/s2-pro/codec.pth \
  --decoder-config-name modded_dac_vq \
  --listen 0.0.0.0:9999
```

> Use port 9999 on Vast.ai — port 8080 conflicts with Cloudflare tunnels on Jupyter templates.

### Step 4 — Synthesize speech

```bash
python src/inference.py \
  --reference output/cleaned/my_recording_clean.wav \
  --text "Your text to synthesize here." \
  --out output/result.wav \
  --server http://localhost:9999
```

---

## Known Issues / Notes

- **Python 3.13 breaks pydub** — `audioop` was removed in 3.13. Use Python 3.11 via conda.
- **Demucs downloads ~80MB of model weights on first run** — make sure you have a stable connection before batch processing.
- **Reference audio quality is everything** — noisy or very short clips produce noticeably worse similarity. Clean, dry, single-speaker audio works best.
- **Vast.ai** — use the PyTorch template, not Jupyter. Run the server on port 9999 and expose it in your instance config.
- **Windows WSL2** — only GPU 0 is accessible for CUDA. Multi-GPU is not supported under WSL.

---

## Ethics

This project is for **educational and research purposes only**.

- The examples in this repo use my own voice as the reference speaker.
- Do not use this pipeline to clone any other person's voice without their explicit, informed consent.
- Do not use this project to download or process audio you don't have the right to use.
- Do not use synthesized audio to impersonate, defraud, or deceive anyone.
- Always disclose when audio is synthetically generated.

If you're not sure whether a use case is okay, assume it isn't.

---

## License

MIT — see [LICENSE](LICENSE).
