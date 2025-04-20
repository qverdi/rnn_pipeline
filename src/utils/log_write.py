import os
import queue
import threading
import pandas as pd

class LogWriter:
    """Background thread that writes logs from the queue to a file."""

    def __init__(self, output_file, log_queue):
        self.log_queue = log_queue
        self.output_file = output_file

        self.stop_event = threading.Event()
        self.writer_thread = threading.Thread(target=self._write_logs)
        self.writer_thread.start()

    def _write_logs(self):
        """Continuously writes logs from queue to file until queue is empty and stop is requested."""
        while not self.stop_event.is_set() or not self.log_queue.empty():
            try:
                row = self.log_queue.get(timeout=0.1)  # Fetch log entry
                if row is None:
                    break  # Stop signal received

                df = pd.DataFrame([row])

                if os.path.exists(self.output_file):
                    df.to_csv(self.output_file, mode="a", index=False, header=False)
                else:
                    df.to_csv(self.output_file, mode="w", index=False, header=True)

            except queue.Empty:
                continue  # Wait until the queue has entries

    def stop(self):
        """Stops the writer thread when the queue is empty."""
        self.stop_event.set()  # Stop signal
        self.log_queue.put(None) 
        self.writer_thread.join()  # Wait until writing is done
