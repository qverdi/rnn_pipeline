import subprocess
from src.config.file_constants import REPORT_FILE

def main():
    # Start Streamlit with the current script in the background
    subprocess.Popen(["streamlit", "run", REPORT_FILE])

if __name__ == "__main__":
    main()
