import cProfile
import pstats


class Profiler:
    def __init__(self):
        self.profiler = cProfile.Profile()

    def __enter__(self):
        self.profiler.enable()
        return self.profiler

    def __exit__(self, exc_type, exc_value, traceback):
        self.profiler.disable()
        stats = pstats.Stats(self.profiler).sort_stats('cumtime')
        stats.print_stats()
