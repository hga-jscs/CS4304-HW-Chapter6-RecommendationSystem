"""Run the unchanged, pinned Synerise validator and three official downstream tasks."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys


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
        code = process.wait()
        if code:
            raise subprocess.CalledProcessError(code, command)


def check_official_code(directory: Path) -> str:
    actual = subprocess.check_output(["git", "-C", str(directory), "rev-parse", "HEAD"], text=True).strip()
    if actual != OFFICIAL_COMMIT:
        raise ValueError(f"Official code must use {OFFICIAL_COMMIT}; found {actual}")
    changed = subprocess.check_output(
        ["git", "-C", str(directory), "status", "--porcelain", "--untracked-files=no"], text=True).strip()
    if changed:
        raise ValueError("Official tracked files have local changes; restore the pinned evaluator first")
    return actual


def evaluate(official_directory: str | Path, data_directory: str | Path,
             embeddings_directory: str | Path, score_directory: str | Path,
             python: str = sys.executable, accelerator: str = "gpu", device: str = "0",
             num_workers: int = 0, validate_only: bool = False, seed: int = 42) -> dict:
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
    check_official_code(official_directory)
    score_directory.mkdir(parents=True, exist_ok=True)
    if any((score_directory / name).exists() for name in ("scores.json", "epochs.json", "evaluation.json")):
        raise FileExistsError("Evaluation results already exist; use a new score directory")
    validation = [python, "-m", "validator.run", "--data-dir", str(data_directory),
                  "--embeddings-dir", str(embeddings_directory)]
    run_logged(validation, official_directory, score_directory / "validation.log")
    commands = [validation]
    if not validate_only:
        if num_workers < 0:
            raise ValueError("num_workers must be non-negative")
        # Only initialize random generators; execute the unchanged official module.
        initialize = ("import sys,runpy; from pytorch_lightning import seed_everything; "
                      "seed_everything(int(sys.argv.pop(1)), workers=True); "
                      "runpy.run_module('training_pipeline.train', run_name='__main__')")
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
    result = {"official_repository": OFFICIAL_REPOSITORY, "official_commit": OFFICIAL_COMMIT,
              "validation_passed": True, "training_completed": not validate_only,
              "tasks": list(TASKS) if not validate_only else [], "seed": seed, "commands": commands}
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
    parser.add_argument("--validate-only", action="store_true")
    print(json.dumps(evaluate(**vars(parser.parse_args())), indent=2))


if __name__ == "__main__":
    main()
