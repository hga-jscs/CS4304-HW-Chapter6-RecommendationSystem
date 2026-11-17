"""Generate a private CPU Kaggle script job from the current local source."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path


COMPETITION = "kkbox-music-recommendation-challenge"
SOURCE_FILES = [
    "basic/__init__.py", "basic/features.py", "basic/train.py",
    "basic/predict.py", "basic/submission.py",
]
RUNTIME_PACKAGES = {"numpy", "pandas", "scipy", "scikit-learn", "lightgbm", "pyarrow", "py7zr"}

# Reuse the Kaggle image by default; installation is an explicit optional mode.
RUNNER = r'''
import hashlib
import importlib
import importlib.metadata
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys
import time

INPUT_ROOT = Path("/kaggle/input")
WORKING_ROOT = Path("/kaggle/working")
DATA_ROOT = Path("/kaggle/temp/kkbox-data")
EXPECTED_FILES = ["train.csv", "test.csv", "members.csv", "songs.csv", "song_extra_info.csv", "sample_submission.csv"]


def check_runtime_dependencies():
    """Fail early with useful diagnostics; never implicitly access the network."""
    modules = {"numpy": "numpy", "pandas": "pandas", "scipy": "scipy",
               "scikit-learn": "sklearn", "lightgbm": "lightgbm", "pyarrow": "pyarrow"}
    versions, failures = {}, []
    for package, module_name in modules.items():
        try:
            importlib.import_module(module_name)
            versions[package] = importlib.metadata.version(package)
        except Exception as error:
            failures.append(f"{package}: {type(error).__name__}: {error}")
    print("Runtime dependency versions:", json.dumps(versions, sort_keys=True), flush=True)
    if failures:
        raise RuntimeError("Missing or unusable Kaggle runtime dependencies; no automatic pip install was attempted. "
                           "Use an image containing the required packages, or explicitly enable --install-dependencies "
                           "when network access is available. " + "; ".join(failures))
    return versions


def select_member(names, filename):
    candidates = [name for name in names if PurePosixPath(name.replace("\\", "/")).name == filename]
    if len(candidates) != 1:
        raise ValueError(f"Archive does not contain exactly one {filename}")
    member = PurePosixPath(candidates[0].replace("\\", "/"))
    if member.is_absolute() or ".." in member.parts or ":" in str(member):
        raise ValueError(f"Invalid archive member path: {member}")
    return candidates[0], member


def extract_archive(archive_path, filename, destination):
    seven_zip = next((path for command in ("7z", "7zz", "7za") if (path := shutil.which(command))), None)
    if seven_zip:
        listing = subprocess.run([seven_zip, "l", "-slt", str(archive_path)],
                                 check=True, capture_output=True, text=True, encoding="utf-8", errors="replace")
        # Technical listing separates archive properties from member properties.
        body = listing.stdout.replace("\r\n", "\n").split("\n----------\n", 1)
        if len(body) != 2:
            raise ValueError("Cannot identify archive members in system 7z listing")
        names = [line[len("Path = "):] for line in body[1].splitlines() if line.startswith("Path = ")]
        member, _ = select_member(names, filename)
        # -so streams exactly the selected member into our controlled destination,
        # so archive path names cannot choose an extraction path.
        with destination.open("wb") as stream:
            subprocess.run([seven_zip, "e", "-so", "-bd", "-y", str(archive_path), member],
                           stdout=stream, stderr=subprocess.PIPE, check=True)
        print(f"Extracted with system 7z: {filename}", flush=True)
        return
    try:
        import py7zr
    except ImportError as error:
        raise RuntimeError("Mounted input contains .7z archives, but neither system 7z/7zz/7za nor py7zr is available. "
                           "Provide extracted official CSVs or an environment with a 7z extractor.") from error
    with py7zr.SevenZipFile(archive_path, mode="r") as archive:
        member_name, member = select_member(archive.getnames(), filename)
        extraction = destination.parent / (filename + "_extracted")
        archive.extract(path=extraction, targets=[member_name])
    extracted = extraction.joinpath(*member.parts)
    if not extracted.is_file():
        raise ValueError(f"Expected extracted CSV was not created: {extracted}")
    extracted.replace(destination)
    print(f"Extracted with py7zr: {filename}", flush=True)


def prepare_data(input_root=INPUT_ROOT, data_root=DATA_ROOT):
    """Use mounted CSVs or extract only the six expected official CSV members."""
    data_root.mkdir(parents=True, exist_ok=True)
    for filename in EXPECTED_FILES:
        destination = data_root / filename
        if destination.exists():
            raise FileExistsError(f"Fresh job expected; data destination already exists: {destination}")
        direct = list(input_root.rglob(filename))
        if len(direct) > 1:
            raise ValueError(f"Ambiguous mounted input for {filename}: {direct}")
        if direct:
            destination.symlink_to(direct[0].resolve())
            print(f"Using mounted CSV: {direct[0]}", flush=True)
            continue
        archives = list(input_root.rglob(filename + ".7z"))
        if len(archives) != 1:
            raise FileNotFoundError(f"Need one official {filename} or {filename}.7z under {input_root}; found {archives}")
        extract_archive(archives[0], filename, destination)
        print(f"Extracted {filename}: {destination.stat().st_size:,} bytes", flush=True)
    return data_root


def run_checked(arguments, cwd, env, log_path):
    """Stream subprocess output to the notebook log and an experiment log."""
    print("Running:", " ".join(map(str, arguments)), flush=True)
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(arguments, cwd=cwd, env=env, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
        return_code = process.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, arguments)


def main():
    started = time.perf_counter()
    WORKING_ROOT.mkdir(parents=True, exist_ok=True)
    print(json.dumps(JOB_SETTINGS, indent=2), flush=True)
    print("Python:", sys.version, "CPU count:", os.cpu_count(), flush=True)
    memory_info = Path("/proc/meminfo")
    if memory_info.exists():
        print("\n".join(line for line in memory_info.read_text().splitlines()
                        if line.startswith(("MemTotal:", "MemAvailable:"))), flush=True)
    if JOB_SETTINGS["install_dependencies"]:
        subprocess.run([sys.executable, "-m", "pip", "install", "--disable-pip-version-check",
                        "--no-input", "--no-cache-dir", *RUNTIME_REQUIREMENTS], check=True)
    actual_versions = check_runtime_dependencies()
    (WORKING_ROOT / "runtime_versions.json").write_text(json.dumps({"python": sys.version,
        "packages": actual_versions}, indent=2)+"\n", encoding="utf-8")
    code_root = WORKING_ROOT / "code"
    for relative_path, content in SOURCES.items():
        destination = code_root / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(content, encoding="utf-8")
    manifest = dict(SOURCE_MANIFEST)
    manifest["job_settings"] = JOB_SETTINGS
    manifest["runtime_requirements"] = RUNTIME_REQUIREMENTS
    manifest["actual_runtime_versions"] = actual_versions
    (WORKING_ROOT / "source_manifest.json").write_text(json.dumps(manifest, indent=2)+"\n", encoding="utf-8")
    data_root = prepare_data()
    env = os.environ.copy()
    env.update({"PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8",
                "OMP_NUM_THREADS": str(JOB_SETTINGS["threads"]),
                "OPENBLAS_NUM_THREADS": str(JOB_SETTINGS["threads"])})
    completed = []
    for feature_mode in JOB_SETTINGS["feature_modes"]:
        output = WORKING_ROOT / feature_mode
        rounds = JOB_SETTINGS[feature_mode + "_rounds"]
        training_command = [sys.executable, "-m", "basic.train", "--data-dir", str(data_root),
            "--output-dir", str(output), "--feature-mode", feature_mode,
            "--num-boost-round", str(rounds), "--early-stopping-rounds", str(JOB_SETTINGS["early_stopping_rounds"]),
            "--threads", str(JOB_SETTINGS["threads"]), "--seed", str(JOB_SETTINGS["seed"]),
            "--max-train-rows", str(JOB_SETTINGS["max_train_rows"])]
        if JOB_SETTINGS["disable_frequency_features"]:
            training_command.append("--disable-frequency-features")
        if JOB_SETTINGS["user_context_features"]:
            training_command.append("--user-context-features")
        run_checked(training_command, code_root, env, WORKING_ROOT / (feature_mode + "_training.log"))
        prediction_command = [sys.executable, "-m", "basic.predict", "--data-dir", str(data_root),
            "--model-dir", str(output), "--output", str(output / "submission.csv"),
            "--threads", str(JOB_SETTINGS["threads"])]
        run_checked(prediction_command, code_root, env, WORKING_ROOT / (feature_mode + "_prediction.log"))
        metrics = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
        completed.append({"feature_mode": feature_mode, "metrics": metrics,
                          "submission": str(output / "submission.csv")})
        (WORKING_ROOT / "experiment_summary.json").write_text(json.dumps({"completed": completed,
            "elapsed_seconds": time.perf_counter()-started}, indent=2)+"\n", encoding="utf-8")
    print("All requested experiments completed and submission files passed validation.", flush=True)


if __name__ == "__main__":
    main()
'''


def generate_job(
    owner: str, slug: str, output_directory: Path, *, feature_modes: list[str] | None = None,
    baseline_rounds: int = 400, enhanced_rounds: int = 600, early_stopping_rounds: int = 30,
    threads: int = 4, seed: int = 42, max_train_rows: int = 0, install_dependencies: bool = False,
    disable_frequency_features: bool = False,
    user_context_features: bool = False,
) -> dict:
    """Write metadata/run.py only; never authenticate, upload, or start a job."""
    if not re.fullmatch(r"[A-Za-z0-9_-]+", owner):
        raise ValueError("owner must be a Kaggle username, not a URL or path")
    if not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", slug) or not 5 <= len(slug) <= 50:
        raise ValueError("slug must contain 5–50 lowercase letters/digits separated by single hyphens")
    modes = feature_modes if feature_modes is not None else ["baseline", "enhanced"]
    if not modes or len(set(modes)) != len(modes) or any(mode not in {"baseline", "enhanced"} for mode in modes):
        raise ValueError("feature_modes must contain baseline and/or enhanced once each")
    if min(baseline_rounds, enhanced_rounds, early_stopping_rounds, threads) <= 0 or max_train_rows < 0:
        raise ValueError("Rounds/threads must be positive and max_train_rows nonnegative")
    output_directory = Path(output_directory)
    if output_directory.exists() and any(output_directory.iterdir()):
        raise ValueError("Choose a new, empty job output directory")
    source_root = Path(__file__).resolve().parent.parent
    sources = {name: (source_root / name).read_text(encoding="utf-8") for name in SOURCE_FILES}
    requirements = []
    for line in (source_root / "basic/requirements.txt").read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and re.split(r"[<>=!~\[]", line, maxsplit=1)[0] in RUNTIME_PACKAGES:
            requirements.append(line)
    if len(requirements) != len(RUNTIME_PACKAGES):
        raise ValueError("Could not identify all seven runtime dependencies")
    settings = {"feature_modes": modes, "baseline_rounds": baseline_rounds, "enhanced_rounds": enhanced_rounds,
        "early_stopping_rounds": early_stopping_rounds, "threads": threads, "seed": seed,
        "max_train_rows": max_train_rows, "install_dependencies": install_dependencies,
        "disable_frequency_features": disable_frequency_features,
        "user_context_features": user_context_features}
    try:
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source_root, text=True, stderr=subprocess.DEVNULL).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=source_root, text=True, stderr=subprocess.DEVNULL).strip())
    except (OSError, subprocess.CalledProcessError):
        revision, dirty = None, None
    manifest = {"git_revision": revision, "working_tree_dirty": dirty,
        "source_sha256": {name: hashlib.sha256(content.encode("utf-8")).hexdigest() for name, content in sources.items()}}
    metadata = {"id": owner + "/" + slug, "title": slug.replace("-", " "),
        "code_file": "run.py", "language": "python", "kernel_type": "script",
        "is_private": True, "enable_gpu": False, "enable_internet": install_dependencies,
        "competition_sources": [COMPETITION], "dataset_sources": [], "kernel_sources": [], "model_sources": []}
    header = "# Generated by basic.kaggle_job; source snapshot is embedded below.\n"
    header += "import json\n"
    for name, value in [("SOURCES", sources), ("RUNTIME_REQUIREMENTS", requirements), ("JOB_SETTINGS", settings), ("SOURCE_MANIFEST", manifest)]:
        header += f"{name} = json.loads({json.dumps(json.dumps(value, ensure_ascii=True))})\n"
    script = header + RUNNER
    compile(script, "run.py", "exec")
    output_directory.mkdir(parents=True, exist_ok=True)
    (output_directory / "run.py").write_text(script, encoding="utf-8")
    (output_directory / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2)+"\n", encoding="utf-8")
    (output_directory / "job_configuration.json").write_text(json.dumps({"settings": settings, "source": manifest}, indent=2)+"\n", encoding="utf-8")
    return {"kernel": metadata["id"], "directory": str(output_directory.resolve()),
        "feature_modes": modes, "max_train_rows": max_train_rows,
        "run_script_sha256": hashlib.sha256(script.encode("utf-8")).hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--owner", required=True)
    parser.add_argument("--slug", required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--feature-modes", choices=["baseline", "enhanced"], nargs="+", default=["baseline", "enhanced"])
    parser.add_argument("--baseline-rounds", type=int, default=400)
    parser.add_argument("--enhanced-rounds", type=int, default=600)
    parser.add_argument("--early-stopping-rounds", type=int, default=30)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train-rows", type=int, default=0)
    parser.add_argument("--disable-frequency-features", action="store_true", help="Pass the frequency-feature ablation flag to every training mode")
    parser.add_argument("--user-context-features", action="store_true", help="Pass the smoothed user/context probability option to every training mode")
    parser.add_argument("--install-dependencies", action="store_true", help="Explicitly enable network and pip installation; default reuses the Kaggle image")
    print(json.dumps(generate_job(**vars(parser.parse_args())), indent=2))


if __name__ == "__main__":
    main()
