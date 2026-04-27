#!/usr/bin/env bash
# setup_and_test.sh
#
# Full setup + end-to-end test for voice-cloning-pipeline.
# Run this once on a fresh machine (Ubuntu / Vast.ai) to:
#   1. Clone fish-speech repo
#   2. Download S2 Pro model weights (~11GB total)
#   3. Install dependencies
#   4. Start the API server
#   5. Generate a short test WAV using a sine-wave reference (no real audio needed)
#
# Usage:
#   chmod +x scripts/setup_and_test.sh
#   ./scripts/setup_and_test.sh
#
# On Vast.ai — expose port 9999 in your instance config before running.
# Warning: downloads ~11GB of model weights and may incur cloud bandwidth/storage costs.

# ── Conda activation (must happen before set -e) ──────────────────────────────
# Source conda's shell hook so `conda activate` works in non-interactive shells.
CONDA_BASE="$(conda info --base 2>/dev/null)"
if [ -z "$CONDA_BASE" ]; then
    echo "[ERROR] conda not found. Install Miniconda first: https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi
# shellcheck disable=SC1091
source "$CONDA_BASE/etc/profile.d/conda.sh"

set -e  # exit on any error after conda is sourced

# Config 

CONDA_ENV="audio311"
FISH_DIR="$HOME/fish-speech"
CHECKPOINTS="$FISH_DIR/checkpoints/s2-pro"
SERVER_PORT=9999
TEST_TEXT="This is a test of the voice cloning pipeline."
TEST_REF="test_reference.wav"
TEST_OUT="test_output.wav"

#Colors

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

info()  { echo -e "${GREEN}[INFO]${NC}  $1"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $1"; }
error() { echo -e "${RED}[ERROR]${NC} $1"; exit 1; }

#  Step 1: System deps

info "Installing system dependencies..."
sudo apt-get update -q
sudo apt-get install -y ffmpeg git curl

#  Step 2: Conda env 

if conda env list | grep -q "^${CONDA_ENV}"; then
    info "Conda env '${CONDA_ENV}' already exists — skipping creation."
else
    info "Creating conda env '${CONDA_ENV}' with Python 3.11..."
    conda create -n "$CONDA_ENV" python=3.11 -y
fi

info "Activating conda env '${CONDA_ENV}'..."
conda activate "$CONDA_ENV"

#  Step 3: Clone fish-speech 

if [ -d "$FISH_DIR" ]; then
    info "fish-speech repo already exists at $FISH_DIR — skipping clone."
else
    info "Cloning fish-speech..."
    git clone https://github.com/fishaudio/fish-speech.git "$FISH_DIR"
fi

cd "$FISH_DIR"

# Install fish-speech deps
info "Installing fish-speech dependencies..."
pip install torch==2.4.0+cu121 torchaudio==2.4.0+cu121 \
    --index-url https://download.pytorch.org/whl/cu121 -q
pip install -e ".[stable]" -q

# Install pipeline deps
pip install demucs pydub numpy scipy httpx -q

#  Step 4: Download S2 Pro weights

if [ -f "$CHECKPOINTS/codec.pth" ] && [ -f "$CHECKPOINTS/tokenizer.json" ]; then
    info "S2 Pro weights already present — skipping download."
else
    info "Downloading S2 Pro model weights (~11GB — this will take a while)..."

    if command -v hf &>/dev/null; then
        hf download fishaudio/s2-pro --local-dir "$CHECKPOINTS"
    else
        info "hf CLI not found — installing huggingface_hub..."
        pip install huggingface_hub -q
        python -c "
from huggingface_hub import snapshot_download
snapshot_download(repo_id='fishaudio/s2-pro', local_dir='$CHECKPOINTS')
print('Download complete.')
"
    fi

    info "Verifying downloaded files..."
    ls -lh "$CHECKPOINTS"
fi

# Step 5: Start API server (background) 

info "Starting S2 Pro API server on port $SERVER_PORT..."

pkill -f "api_server.py" 2>/dev/null || true
sleep 1

python tools/api_server.py \
    --llama-checkpoint-path "$CHECKPOINTS" \
    --decoder-checkpoint-path "$CHECKPOINTS/codec.pth" \
    --decoder-config-name modded_dac_vq \
    --listen "0.0.0.0:$SERVER_PORT" \
    --half &

SERVER_PID=$!
info "Server PID: $SERVER_PID — waiting for it to come up..."

for i in $(seq 1 30); do
    if curl -s "http://localhost:$SERVER_PORT" >/dev/null 2>&1; then
        info "Server is up!"
        break
    fi
    echo -n "."
    sleep 2
done
echo ""

# Step 6: Generate a test reference WAV 

info "Generating a synthetic 5s reference WAV (sine wave)..."
python3 - <<'EOF'
import numpy as np, wave, os

sample_rate = 44100
duration    = 5
freq        = 220.0

t      = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
signal = (np.sin(2 * np.pi * freq * t) * 32767 * 0.5).astype(np.int16)

with wave.open("test_reference.wav", "w") as wf:
    wf.setnchannels(1)
    wf.setsampwidth(2)
    wf.setframerate(sample_rate)
    wf.writeframes(signal.tobytes())

print(f"Saved test_reference.wav ({os.path.getsize('test_reference.wav') / 1024:.1f} KB)")
EOF

#  Step 7: Run inference test 

info "Running inference test..."

python3 - <<EOF
import httpx, sys

endpoint = "http://localhost:${SERVER_PORT}/v1/tts"
text     = "${TEST_TEXT}"
ref_path = "${TEST_REF}"
out_path = "${TEST_OUT}"

try:
    with open(ref_path, "rb") as ref_f:
        response = httpx.post(
            endpoint,
            data={"text": text},
            files={"reference": (ref_path, ref_f, "audio/wav")},
            timeout=120.0,
        )

    if response.status_code == 200:
        with open(out_path, "wb") as f:
            f.write(response.content)
        print(f"✓ SUCCESS — output saved: {out_path} ({len(response.content)/1024:.1f} KB)")
        sys.exit(0)
    else:
        print(f"✗ Server returned {response.status_code}: {response.text}")
        sys.exit(1)

except httpx.ConnectError:
    print("✗ Could not connect to server — it may still be loading. Try again in 30s.")
    sys.exit(1)
EOF

STATUS=$?

#  Done 

echo ""
if [ $STATUS -eq 0 ]; then
    info "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    info "All tests passed."
    info "Output WAV : $FISH_DIR/$TEST_OUT"
    info "Server PID : $SERVER_PID (still running)"
    info "  To stop  : kill $SERVER_PID"
    info "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
else
    warn "Inference test failed — server is still running (PID $SERVER_PID)."
    warn "Check server logs above, wait a bit longer for model load, then retry:"
    warn "  python src/inference.py --reference $TEST_REF --text \"$TEST_TEXT\" --out $TEST_OUT"
fi
