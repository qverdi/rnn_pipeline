import multiprocessing
import threading
import os
import pandas as pd
import time
from pathlib import Path
from queue import Empty  # Explicitly catch empty queue

from src.config.file_constants import OUTPUT_DIR

class LogPost:
    def __init__(self, log: dict, log_queue: multiprocessing.Queue):
        """
        Logger that uses a shared multiprocessing queue to safely log from multiple processes.

        Args:
            log (dict): Metadata for logging.
            log_queue (multiprocessing.Queue): Shared queue for logging.
        """
        self.log = log
        self.log_queue = log_queue


    def append_results(self, should_evaluate, model_metrics, budget):
        """Push results into the logging queue instead of writing directly."""
        row = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            **self.log,
            "budget": budget,
            "evaluation_epoch": should_evaluate,
            **model_metrics.to_dict(),
        }
        self.log_queue.put(row)
