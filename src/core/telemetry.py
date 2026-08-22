import time
from typing import Dict

class LatencyProfiler:
    """High-precision latency profiler for retrieval microbenchmarks."""
    def __init__(self):
        self.timers: Dict[str, float] = {}
        self._start_times: Dict[str, float] = {}

    def start(self, name: str):
        self._start_times[name] = time.perf_counter()

    def stop(self, name: str):
        if name in self._start_times:
            elapsed = (time.perf_counter() - self._start_times[name]) * 1000
            self.timers[name] = round(elapsed, 3)

    def get_summary(self) -> Dict[str, float]:
        return self.timers
