from pathlib import Path
import json
import yaml
import subprocess
import hashlib

PROJECT_ROOT = Path(__file__).resolve().parent.parent

EXCLUDED_DIRECTORIES = {".git", "my_env", "data", "__pycache__"}
EXCLUDED_FILES = {".env"}


def get_safe_path(path):
    requested_path = Path(path)

    try:
        resolved_path = (PROJECT_ROOT / requested_path).resolve()
        relative_path = resolved_path.relative_to(PROJECT_ROOT)
    except ValueError:
        return None

    if relative_path.name in EXCLUDED_FILES:
        return None

    if any(part in EXCLUDED_DIRECTORIES for part in relative_path.parts):
        return None

    return resolved_path


def list_files():
    files = []

    for path in PROJECT_ROOT.rglob("*"):
        relative_path = path.relative_to(PROJECT_ROOT)

        if get_safe_path(relative_path) is None:
            continue

        if path.is_file():
            files.append(str(relative_path))

    return "\n".join(sorted(files))


def read_file(path):
    file_path = get_safe_path(path)

    if file_path is None:
        return "Access denied."

    if not file_path.exists():
        return f"File not found: {path}"

    if not file_path.is_file():
        return f"Not a file: {path}"

    return file_path.read_text(encoding="utf-8")

def write_file(path, content):
    file_path = get_safe_path(path)

    if file_path is None:
        return "Access denied."

    if file_path.exists() and not file_path.is_file():
        return f"Not a file: {path}"

    file_path.write_text(content, encoding="utf-8")
    return f"Successfully wrote to {path}."

def git_diff():
    try:
        result = subprocess.run(
            ["git", "diff"],
            cwd=PROJECT_ROOT,
            text=True,
            capture_output=True
        )

        if result.returncode != 0:
            return f"Git diff failed: {result.stderr}"

        if not result.stdout:
            return "No uncommitted changes."

        return result.stdout

    except Exception as e:
        return f"Failed to get git diff: {e}"

def list_experiments():
    experiments_path = PROJECT_ROOT / "experiments"

    if not experiments_path.exists():
        return "No experiments directory found."

    experiment_directories = sorted(
        [
            path
            for path in experiments_path.iterdir()
            if path.is_dir() and path.name.startswith("EXP-")
        ]
    )

    if not experiment_directories:
        return "No experiments found."

    summaries = []

    for experiment_path in experiment_directories:
        experiment_id = experiment_path.name
        config_file = experiment_path / "config.yaml"
        results_file = experiment_path / "results.json"

        summary = [experiment_id]

        if config_file.exists():
            with config_file.open("r", encoding="utf-8") as file:
                config = yaml.safe_load(file) or {}

            experiment_config = config.get("experiment", {})
            data_config = config.get("data", {})
            model_config = config.get("model", {})
            training_config = config.get("training", {})

            objective = experiment_config.get("objective")
            code_changes = experiment_config.get("code_changes")
            model = model_config.get("name")
            total_params = model_config.get("total_params")
            epochs = training_config.get("epochs")
            batch_size = training_config.get("batch_size")
            learning_rate = training_config.get("learning_rate")
            n_splits = data_config.get("n_splits")

            if objective:
                summary.append(f"  Objective: {objective}")

            if code_changes:
                summary.append(f"  Code changes: {code_changes}")

            if model:
                summary.append(f"  Model: {model}")

            if total_params:
                summary.append(f"  Parameters: {total_params:,}")

            if epochs is not None:
                summary.append(f"  Epochs: {epochs}")

            if batch_size is not None:
                summary.append(f"  Batch size: {batch_size}")

            if learning_rate is not None:
                summary.append(f"  Learning rate: {learning_rate}")

            if n_splits is not None:
                summary.append(f"  Folds: {n_splits}")

        if results_file.exists():
            with results_file.open("r", encoding="utf-8") as file:
                results = json.load(file)

            aggregate_metrics = results.get("aggregate_test_metrics", {})

            if "accuracy_mean" in aggregate_metrics:
                summary.append(
                    f"  Accuracy: {aggregate_metrics['accuracy_mean']:.4f}"
                )

            if "macro_f1_mean" in aggregate_metrics:
                summary.append(
                    f"  Macro F1: {aggregate_metrics['macro_f1_mean']:.4f}"
                )

            if "experiment_elapsed_time" in results:
                summary.append(
                    f"  Time: {results['experiment_elapsed_time']}"
                )

        summaries.append("\n".join(summary))

    return "\n\n".join(summaries)

# Helper function for creating new EXP ID
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
def get_next_experiment_id():
    experiment_dirs = [
        d for d in EXPERIMENTS_DIR.iterdir()
        if d.is_dir() and d.name.startswith("EXP-")
    ]

    if not experiment_dirs:
        return "EXP-001"

    experiment_numbers = []
    for d in experiment_dirs:
        try:
            experiment_numbers.append(int(d.name.split("-")[1]))
        except (IndexError, ValueError):
            continue

    next_number = max(experiment_numbers, default=0) + 1
    return f"EXP-{next_number:03d}"

def run_experiment(objective=None, code_changes=None, epochs=None, learning_rate=None, batch_size=None, model_name=None, scheduler=None):
    experiment_id = get_next_experiment_id()

    command = ["bash", "scripts/run_experiment.sh", "configs/baseline.yaml", "--experiment", experiment_id]

    if objective is not None:
        command.extend(["--objective", objective])
    if epochs is not None:
        command.extend(["--epochs", str(epochs)])
    if learning_rate is not None:
        command.extend(["--learning-rate", str(learning_rate)])
    if batch_size is not None:
        command.extend(["--batch-size", str(batch_size)])
    if model_name is not None:
        command.extend(["--model", model_name])
    if scheduler is not None:
        command.extend(["--scheduler", scheduler])

    print(f"\nStarting experiment: {experiment_id}")

    try:
        result = subprocess.run(command, cwd=PROJECT_ROOT, text=True)

        if result.returncode != 0:
            return f"Experiment {experiment_id} failed with exit code {result.returncode}."

        # So the agent knows what change was implemented for each Exp
        if code_changes is not None:
            config_path = EXPERIMENTS_DIR / experiment_id / "config.yaml"

            with config_path.open("r", encoding="utf-8") as file:
                config = yaml.safe_load(file) or {}

            config["experiment"]["code_changes"] = code_changes

            with config_path.open("w", encoding="utf-8") as file:
                yaml.dump(config, file, sort_keys=False)

        return f"Experiment {experiment_id} completed successfully."

    except Exception as e:
        return f"Failed to start experiment {experiment_id}: {e}"
    
