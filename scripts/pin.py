#!/usr/bin/env python3
"""Pin the package to one revision of the model on Hugging Face.

    python3 scripts/pin.py <commit>               # rewrites Sources/WhisperANE/Pins.swift
    python3 scripts/pin.py <commit> --json p.json # also writes the pin as JSON (whisper-ane --pin p.json)

Sizes and SHA-256 come from the Hugging Face tree API for files stored with LFS. Every other file is downloaded
and hashed here. Only the Python standard library is used.
"""
import argparse
import hashlib
import json
import pathlib
import re
import sys
import urllib.parse
import urllib.request

REPO = "mashbox-labs/whisper-large-v3-turbo-coreml-ane"
SKIP = {".gitattributes", "README.md"}
SKIP_DIRS = ("assets/",)  # model card images, not needed to run
NEEDED = {"manifest.json", "vocab.json", "embed_tokens.f16", "embed_positions.f16"}
PACKAGES = ("MelSpectrogram.mlmodelc/", "AudioEncoder.mlmodelc/", "TextDecoder.mlmodelc/")


def get(url):
    req = urllib.request.Request(url, headers={"User-Agent": "whisper-ane-pin"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read()


def tree(repo, rev):
    url = f"https://huggingface.co/api/models/{repo}/tree/{rev}?recursive=true&expand=true"
    return [e for e in json.loads(get(url)) if e.get("type") == "file"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("revision", help="full 40-character commit hash")
    ap.add_argument("--repo", default=REPO)
    ap.add_argument("--out", default=str(pathlib.Path(__file__).resolve().parent.parent / "Sources/WhisperANE/Pins.swift"))
    ap.add_argument("--json", help="also write the pin to this JSON file")
    ap.add_argument("--no-swift", action="store_true", help="only write --json")
    a = ap.parse_args()
    if not re.fullmatch(r"[0-9a-f]{40}", a.revision):
        sys.exit("revision must be a full 40-character commit hash")

    files = []
    for e in sorted(tree(a.repo, a.revision), key=lambda e: e["path"]):
        path = e["path"]
        if path in SKIP or path.startswith(SKIP_DIRS):
            continue
        lfs = e.get("lfs")
        if lfs:
            sha, size = lfs["oid"], int(lfs["size"])
        else:
            body = get(f"https://huggingface.co/{a.repo}/resolve/{a.revision}/{urllib.parse.quote(path)}")
            sha, size = hashlib.sha256(body).hexdigest(), len(body)
            if size != int(e["size"]):
                sys.exit(f"{path}: size {size} != tree size {e['size']}")
            if path == "manifest.json":
                m = json.loads(body)
                print(f"manifest: buckets {m.get('buckets')}, one_program {m.get('one_program', False)}", file=sys.stderr)
        files.append({"path": path, "size": size, "sha256": sha})

    paths = {f["path"] for f in files}
    missing = sorted(NEEDED - paths) + [p for p in PACKAGES if not any(x.startswith(p) for x in paths)]
    if missing:
        sys.exit(f"revision is missing {missing}")

    pin = {"repo": a.repo, "revision": a.revision, "files": files}
    if a.json:
        pathlib.Path(a.json).write_text(json.dumps(pin, indent=1) + "\n")
    if not a.no_swift:
        rows = "\n".join(f'            .init(path: "{f["path"]}", size: {f["size"]}, sha256: "{f["sha256"]}"),' for f in files)
        pathlib.Path(a.out).write_text(f"""// Written by scripts/pin.py. Do not edit by hand.

extension ModelPin {{
    /// The model revision this version of the package loads.
    public static let current = ModelPin(
        repo: "{a.repo}",
        revision: "{a.revision}",
        files: [
{rows}
        ]
    )
}}
""")
    total = sum(f["size"] for f in files)
    print(f"pinned {a.repo}@{a.revision}: {len(files)} files, {total:,} bytes", file=sys.stderr)


if __name__ == "__main__":
    main()
