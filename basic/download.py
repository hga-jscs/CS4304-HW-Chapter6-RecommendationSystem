"""Download official KKBox data (requires accepted rules and local Kaggle credentials)."""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from zipfile import ZipFile

from basic.submission import COMPETITION, file_sha256

FILES = ("train.csv", "test.csv", "members.csv", "songs.csv", "song_extra_info.csv", "sample_submission.csv")


def prepare_data(data_dir: Path, archive: Path | None = None, local_only: bool = False) -> dict:
    data_dir.mkdir(parents=True, exist_ok=True)
    if archive is None:
        archive = data_dir / f"{COMPETITION}.zip"
    needs_download = any(not (data_dir / name).exists() and not (data_dir / (name + ".7z")).exists() for name in FILES)
    if needs_download and not archive.exists() and not local_only:
        subprocess.run([sys.executable, "-m", "kaggle", "competitions", "download",
                        "-c", COMPETITION, "-p", str(data_dir.resolve())], check=True)
    if archive.exists():
        with ZipFile(archive) as compressed:
            for member in compressed.infolist():
                name = Path(member.filename).name
                if name not in FILES and name not in tuple(f + ".7z" for f in FILES):
                    continue
                destination = data_dir / name
                if not destination.exists():
                    temporary = destination.with_name(destination.name + ".partial")
                    with compressed.open(member) as source, temporary.open("wb") as target:
                        shutil.copyfileobj(source, target)
                    temporary.replace(destination)
    for name in FILES:
        destination = data_dir / name
        if destination.exists():
            continue
        compressed_path = data_dir / (name + ".7z")
        if not compressed_path.exists():
            raise FileNotFoundError(f"Missing {name} or {name}.7z in {data_dir}")
        import py7zr
        print(f"Extracting {compressed_path.name}", flush=True)
        with tempfile.TemporaryDirectory(prefix="extract_", dir=data_dir) as temporary:
            with py7zr.SevenZipFile(compressed_path) as compressed:
                members = compressed.getnames()
                if members != [name]:
                    raise ValueError(f"Unexpected archive members in {compressed_path.name}: {members}")
                compressed.extractall(path=temporary)
            (Path(temporary) / name).replace(destination)
    manifest = {"competition": COMPETITION, "files": {}}
    for name in FILES:
        path = data_dir / name
        manifest["files"][name] = {"bytes": path.stat().st_size, "sha256": file_sha256(path)}
    (data_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--archive", type=Path, help="An already downloaded official ZIP")
    parser.add_argument("--local-only", action="store_true", help="Extract existing files without authenticating")
    args = parser.parse_args()
    print(json.dumps(prepare_data(args.data_dir, args.archive, args.local_only), indent=2))


if __name__ == "__main__":
    main()
