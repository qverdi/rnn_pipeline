from src.experiments.experiment_handler import ExperimentHandler
from src.utils.input_validator import InputValidator
from src.utils.gpu_config import configure_gpu
import src.utils.git_commands as git

def main():
    """
    Orchestrates the workflow for running an experiment.

    Workflow:
    1. Parses command-line arguments (`--push` and `--name`).
    2. Generates an experiment branch name:
       - If `--name` is provided: `experiment/DDMMYYYY_experiment_name`
       - Otherwise: `experiment/DDMMYYYY`
    3. Configures GPU settings using `configure_gpu()`.
    4. Validates input parameters using `InputValidator`.
    5. Initializes and executes the experiment via `ExperimentHandler`.
    6. If `--push` is provided, commits and pushes all changes to Git.
    7. If `--push` is provided, updates deploy branch with new experiment branch

    Classes Used:
    - InputValidator: Ensures input parameters are valid before execution.
    - ExperimentHandler: Manages the full experiment lifecycle, including execution.

    Example Usage:
        poetry run experiment --push --name my_experiment
        poetry run experiment --push  # Uses only the date
    """
    args = git.parse_args()
    branch_name = git.get_branch_name(args.name)

    print(f"🚀 Running experiment: {args.name or '(No Name)'}")

    configure_gpu()

    input_validator = InputValidator()
    input_validator.validate()

    experiment_handler = ExperimentHandler()
    experiment_handler.conduct()

    if args.push:
        print(f"📌 Pushing experiment to Git branch: {branch_name}")
        git.git_commit_and_push_experiment(branch_name)


if __name__ == "__main__":
    main()
