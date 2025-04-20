import argparse
import subprocess
from datetime import datetime
from src.utils.os_utils import update_readme

def parse_args():
    """
    Parse command-line arguments for running an experiment.

    Arguments:
        --push (optional): If provided, commits and pushes experiment changes to Git.
        --name (optional): Name of the experiment (appended to branch name if given).

    Returns:
        argparse.Namespace: Parsed arguments with attributes `push` (bool) and `name` (str or None).
    """
    parser = argparse.ArgumentParser(description="Run an experiment and optionally push results.")
    parser.add_argument("--push", action="store_true", help="Commit and push experiment changes.")
    parser.add_argument("--name", type=str, help="Optional name of the experiment (appended to branch name).")

    return parser.parse_args()


def force_update_deploy_from_experiment(branch_name):
    """
    Forces the 'deploy' branch to be a copy of the experiment branch.

    Args:
        branch_name (str): The experiment branch name.

    Raises:
        subprocess.CalledProcessError: If any Git command fails.
    """
    try:
        print("🔄 Switching to 'deploy' branch...")
        subprocess.run(["git", "switch", "deploy"], check=True)
        subprocess.run(["git", "pull", "origin", "deploy"], check=True)

        print(f"⚡ Overwriting 'deploy' with '{branch_name}'...")
        subprocess.run(["git", "reset", "--hard", branch_name], check=True)

        print("🚀 Force pushing to 'deploy' branch...")
        subprocess.run(["git", "push", "--force", "origin", "deploy"], check=True)
        print("✅ Deploy branch is now an exact copy of the experiment branch!")

    except subprocess.CalledProcessError as e:
        print(f"❌ Git operation failed: {e}")

def git_commit_and_push_experiment(branch_name):
    """
    Creates a new Git branch for the experiment, commits all changes, and pushes it.
    Merges new branch into deploy branch, which deploys experiment results to streamlit.
    Updates README.md about streamlit experiment branch information and pushes to main.

    Args:
        branch_name (str): The name of the experiment branch.

    Raises:
        subprocess.CalledProcessError: If any Git command fails.
    """
    try:
        subprocess.run(["git", "branch", branch_name], check=True)
        subprocess.run(["git", "switch", branch_name], check=True)
        subprocess.run(["git", "add", "."], check=True)
        subprocess.run(["git", "commit", "-m", f"init: {branch_name}"], check=True)
        subprocess.run(["git", "push", "-u", "origin", branch_name], check=True)
        print(f"✅ Successfully pushed experiment branch: {branch_name}")

        # Update readme experiment deployment information (main)
        update_readme_and_push(branch_name)
        
        # Deploy new experiment by force update (deploy)
        force_update_deploy_from_experiment(branch_name)

        # Switch to main branch
        subprocess.run(["git", "switch", "main"], check=True)

    except subprocess.CalledProcessError as e:
        print(f"❌ Git operation failed: {e}")

def get_branch_name(name):
    # Generate experiment branch name with date format DDMMYYYY-HHMM
    timestamp = datetime.now().strftime("%d%m%Y-%H%M")  # Format: DDMMYYYY-HHMM
    branch_name = f"experiment/{timestamp}" if not name else f"experiment/{timestamp}_{name}"
    
    return branch_name

def update_readme_and_push(branch_name):
    """
    Updates the README file to indicate the current experiment branch,
    then commits and pushes the changes to the main branch.

    Args:
        branch_name (str): The name of the experiment branch.
    """
    try:

        subprocess.run(["git", "switch", "main"], check=True)

        # Update the README file
        update_readme(branch_name)

        # Stage the updated README file
        subprocess.run(["git", "add", "README.md"], check=True)

        # Commit the changes
        commit_message = f"update: README.md deployed experiment -> {branch_name}"
        subprocess.run(["git", "commit", "-m", commit_message], check=True)

        # Push the changes to the main branch
        subprocess.run(["git", "push", "origin", "main"], check=True)

        print(f"🚀 Successfully updated README and pushed to 'main' branch!")

    except subprocess.CalledProcessError as e:
        print(f"❌ Git operation failed: {e}")


def switch_deployed_experiment(branch_name):
    """
    Updates the 'deploy' branch to match the specified experiment branch 
    and updates the README accordingly.

    Args:
        branch_name (str): The experiment branch to deploy.

    Steps:
    - Force updates 'deploy' to be an exact copy of the experiment.
    - Updates the README and pushes the changes.

    Example:
        switch_deployed_experiment("experiment/14022025-1531_deploy-test")
    """
    force_update_deploy_from_experiment(branch_name)
    update_readme_and_push(branch_name)
