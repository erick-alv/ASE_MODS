

import tensorboard as tb
import pandas as pd
from tensorboard.backend.event_processing import event_accumulator
import os

class ResultsPlotter:
    def __init__(self, config):
        self.test_results_folder = config["test_results_folder"]
        event_acc = event_accumulator.EventAccumulator(os.path.join(self.test_results_folder, "_12-20-49-49"))
        event_acc.Reload()
        mnames = event_acc.Tags()
        print(mnames)


if __name__ == "__main__":
    config = {
        "test_results_folder": "output/HumanoidImitation_30-16-51-12_test_results"
    }
    rp = ResultsPlotter(config)
