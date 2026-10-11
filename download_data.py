"""Download AEGIS JSONL files directly from their Hugging Face dataset repository."""
from pathlib import Path
import argparse
import ssl

import certifi
from urllib.request import Request, urlopen

URLS = {
    "test": "https://huggingface.co/datasets/Fancylalala/AEGIS/resolve/main/Aegis-Bench/test.jsonl",
    "val": "https://huggingface.co/datasets/Fancylalala/AEGIS/resolve/main/Aegis-Training/val.jsonl",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("split", choices=["test", "val"])
    args = ap.parse_args()
    dest = Path(__file__).parent / "data" / f"{args.split}.jsonl"
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 10000:
        print(f"Already downloaded: {dest} ({dest.stat().st_size:,} bytes)")
        return
    partial = dest.with_suffix(".jsonl.partial")
    print(f"Downloading {URLS[args.split]}\nTo: {dest}")
    req = Request(URLS[args.split], headers={"User-Agent": "Aperture-hackathon/1.0"})
    try:
        with urlopen(req, timeout=90, context=ssl.create_default_context(cafile=certifi.where())) as response, partial.open("wb") as out:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                out.write(chunk)
    except Exception:
        partial.unlink(missing_ok=True)
        raise
    partial.replace(dest)
    print(f"Saved: {dest} ({dest.stat().st_size:,} bytes)")

if __name__ == "__main__":
    main()
