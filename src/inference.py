"""
inference.py

Sends a text synthesis request to a locally running Fish Audio S2 Pro
API server, injecting a reference audio file for voice cloning.

The server must be started separately — see README for the launch command.

Usage:
    python src/inference.py \
        --reference output/cleaned_reference.wav \
        --text "Hello, this is a test." \
        --out output/result.wav \
        --server http://localhost:9999
"""

import argparse
import sys
import httpx


def synthesize(
    text: str,
    reference_wav: str,
    output_path: str,
    server_url: str = "http://localhost:9999",
) -> bool:
    """
    Send a TTS request to the local S2 Pro API server with reference audio.

    The server expects a multipart POST to /v1/tts with:
      - 'text'      : the text to synthesize
      - 'reference' : the reference .wav file (binary)

    Returns True on success, False on failure.
    """
    endpoint = f"{server_url.rstrip('/')}/v1/tts"

    print(f"→ Sending request to {endpoint}")
    print(f"  Text      : {text[:80]}{'...' if len(text) > 80 else ''}")
    print(f"  Reference : {reference_wav}")

    try:
        with open(reference_wav, "rb") as ref_f:
            files   = {"reference": (reference_wav, ref_f, "audio/wav")}
            payload = {"text": text}

            response = httpx.post(
                endpoint,
                data=payload,
                files=files,
                timeout=120.0,   # S2 Pro can be slow on first inference
            )

        if response.status_code != 200:
            print(f"✗ Server returned {response.status_code}: {response.text}")
            return False

        with open(output_path, "wb") as out_f:
            out_f.write(response.content)

        size_kb = len(response.content) / 1024
        print(f"✓ Saved {size_kb:.1f} KB → {output_path}")
        return True

    except httpx.ConnectError:
        print(f"✗ Could not connect to server at {server_url}")
        print("  Make sure the S2 Pro API server is running (see README).")
        return False
    except FileNotFoundError:
        print(f"✗ Reference file not found: {reference_wav}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Synthesize speech via local S2 Pro server")
    parser.add_argument("--reference", required=True, help="Path to reference .wav file")
    parser.add_argument("--text",      required=True, help="Text to synthesize")
    parser.add_argument("--out",       required=True, help="Output .wav path")
    parser.add_argument("--server",    default="http://localhost:9999",
                        help="S2 Pro API server URL (default: http://localhost:9999)")
    args = parser.parse_args()

    ok = synthesize(
        text=args.text,
        reference_wav=args.reference,
        output_path=args.out,
        server_url=args.server,
    )
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
