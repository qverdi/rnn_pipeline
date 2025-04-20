import src.utils.git_commands as git
import argparse

def main():
    """
    Parses command-line arguments and deploys the specified experiment branch.

    This script:
    - Takes the experiment branch name as an argument.
    - Calls `switch_deployed_experiment` to update the 'deploy' branch.

    Usage:
        python script.py --name experiment/14022025-1531_deploy-test

    Arguments:
        --name (str): The name of the experiment branch to deploy.

    Raises:
        SystemExit: If no branch name is provided.
    """
    parser = argparse.ArgumentParser(description="Deploy a specified experiment branch.")
    parser.add_argument(
        "--name", 
        type=str, 
        required=True, 
        help="Name of the experiment branch to deploy (e.g., experiment/14022025-1531_deploy-test)."
    )
    
    args = parser.parse_args()

    if not args.name:
        print("❌ Error: You must provide a branch name using --name.")
        parser.print_help()
        exit(1)

    git.switch_deployed_experiment(args.name)

if __name__ == "__main__":
    main()
