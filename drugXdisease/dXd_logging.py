from datetime import datetime
import time

def log_performance(start_time):
    """
    Log performance metrics including compute time to a file.
    """
    end_time = time.time()
    elapsed_time = end_time - start_time
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(f"performance_{timestamp}.txt", "w") as perf_file:
        perf_file.write(f"Elapsed time: {elapsed_time:.2f} seconds\n")