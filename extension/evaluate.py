"""Run pinned Synerise evaluation, optionally caching equivalent downstream targets."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import warnings


OFFICIAL_REPOSITORY = "https://github.com/Synerise/recsys2025.git"
OFFICIAL_COMMIT = "e91c1a62f84611151a399ce157e6031a4a73618f"
TASKS = ("churn", "propensity_category", "propensity_sku")


def run_logged(command: list[str], directory: Path, log_path: Path) -> None:
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(command, cwd=directory, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True,
                                   encoding="utf-8", errors="replace")
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
        code = process.wait()
        if code:
            raise subprocess.CalledProcessError(code, command)


def check_official_code(directory: Path, allow_modified: bool = False) -> dict:
    actual = subprocess.check_output(["git", "-C", str(directory), "rev-parse", "HEAD"], text=True).strip()
    changed = subprocess.check_output(
        ["git", "-C", str(directory), "status", "--porcelain", "--untracked-files=no"], text=True)
    matches_reference = actual == OFFICIAL_COMMIT and not changed
    if not matches_reference:
        message = ("Evaluator differs from the pinned official source. "
                   "Use --allow-modified-official for a labelled debugging run.")
        if not allow_modified:
            raise ValueError(message)
        warnings.warn(message + " Results are not marked as official-protocol verified.")
    return {"commit": actual, "tracked_changes": changed.splitlines(),
            "matches_reference": matches_reference}


def evaluate(official_directory: str | Path, data_directory: str | Path,
             embeddings_directory: str | Path, score_directory: str | Path,
             python: str = sys.executable, accelerator: str = "gpu", device: str = "0",
             num_workers: int = 0, validate_only: bool = False, seed: int = 42,
             cache_targets: bool = False, overwrite: bool = False,
             allow_modified_official: bool = False) -> dict:
    official_directory = Path(official_directory).resolve()
    data_directory = Path(data_directory).resolve()
    embeddings_directory = Path(embeddings_directory).resolve()
    score_directory = Path(score_directory).resolve()
    interpreter = Path(python)
    if interpreter.is_file():
        python = str(interpreter.resolve())
    else:
        located = shutil.which(python)
        if located is None:
            raise FileNotFoundError(f"Python interpreter was not found: {python}")
        python = str(Path(located).resolve())
    official_state = check_official_code(official_directory, allow_modified_official)
    score_directory.mkdir(parents=True, exist_ok=True)
    result_names = ("scores.json", "epochs.json", "evaluation.json")
    if not overwrite and any((score_directory / name).exists() for name in result_names):
        raise FileExistsError("Evaluation results already exist; use --overwrite to replace this experiment")
    if overwrite:
        for name in result_names:
            (score_directory / name).unlink(missing_ok=True)
    validation = [python, "-m", "validator.run", "--data-dir", str(data_directory),
                  "--embeddings-dir", str(embeddings_directory)]
    run_logged(validation, official_directory, score_directory / "validation.log")
    commands = [validation]
    if not validate_only:
        if num_workers < 0:
            raise ValueError("num_workers must be non-negative")
        # Initialize random generators and optionally install the equivalent loader.
        initialize = ("import sys,runpy; from pytorch_lightning import seed_everything; "
                      "seed_everything(int(sys.argv.pop(1)), workers=True); ")
        if cache_targets:
            project_root = str(Path(__file__).resolve().parents[1])
            initialize += (f"sys.path.insert(0, {project_root!r}); "
                           "from extension.target_cache import install; install(); ")
        initialize += "runpy.run_module('training_pipeline.train', run_name='__main__')"
        training = [python, "-c", initialize, str(seed), "--data-dir", str(data_directory),
                    "--embeddings-dir", str(embeddings_directory), "--tasks", *TASKS,
                    "--log-name", score_directory.name, "--accelerator", accelerator,
                    "--devices", "auto" if accelerator == "cpu" else device,
                    "--num-workers", str(num_workers), "--score-dir", str(score_directory)]
        run_logged(training, official_directory, score_directory / "training.log")
        scores = json.loads((score_directory / "scores.json").read_text(encoding="utf-8"))
        epochs = json.loads((score_directory / "epochs.json").read_text(encoding="utf-8"))
        if any(task not in scores or task not in epochs or not math.isfinite(scores[task]) for task in TASKS):
            raise ValueError("Official score files do not contain all three finite task results")
        commands.append(training)
    result = {"official_repository": OFFICIAL_REPOSITORY, "official_commit": official_state["commit"],
              "official_reference_commit": OFFICIAL_COMMIT,
              "official_tracked_changes": official_state["tracked_changes"],
              "official_protocol_verified": official_state["matches_reference"],
              "validation_passed": True, "training_completed": not validate_only,
              "tasks": list(TASKS) if not validate_only else [], "seed": seed,
              "target_cache_enabled": cache_targets, "commands": commands}
    (score_directory / "evaluation.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official-directory", type=Path, required=True)
    parser.add_argument("--data-directory", type=Path, required=True)
    parser.add_argument("--embeddings-directory", type=Path, required=True)
    parser.add_argument("--score-directory", type=Path, required=True)
    parser.add_argument("--python", default=sys.executable, help="Python in the official evaluation environment")
    parser.add_argument("--accelerator", choices=("cpu", "gpu"), default="gpu")
    parser.add_argument("--device", default="0")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cache-targets", action="store_true",
                        help="Cache equivalent official targets in the downstream data loader only")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--allow-modified-official", action="store_true",
                        help="Allow a debugging evaluator and record it as non-reference")
    print(json.dumps(evaluate(**vars(parser.parse_args())), indent=2))


if __name__ == "__main__":
    main()
