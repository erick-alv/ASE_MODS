

import tensorboard as tb
import pandas as pd
from tensorboard.backend.event_processing import event_accumulator
import os
import re

class ResultsPlotter:
    def __init__(self, groups):
        self.test_results_folder = config["test_results_folder"]
        event_acc = event_accumulator.EventAccumulator(os.path.join(self.test_results_folder, "_12-20-49-49"))
        event_acc.Reload()
        mnames = event_acc.Tags()
        print(mnames)

    def load_events(self, events_files_lists):
        return [event_accumulator.EventAccumulator(f) for f in events_files_lists]




def remove_run_suffix(folder_name, is_test_results=False):
    #TODO once we always use the month use regey that just recognize
    # once suffix correct regex is NAME_SUF_REGEX = '_[0-9][0-9](-[0-9][0-9]){4}'
    # or NAME_SUF_REGEX = '_[0-9][0-9]-[0-9][0-9]-[0-9][0-9]-[0-9][0-9]-[0-9][0-9]'
    NAME_SUF_REGEX = '_[0-9][0-9](-[0-9][0-9]){3,4}'
    if is_test_results:
        NAME_SUF_REGEX += '_test_results'

    re_match = re.search(NAME_SUF_REGEX + '$', folder_name)
    if re_match is None:
        raise Exception("Folder name does not match regex")
    ids = re_match.span()
    return folder_name[:ids[0]]


def create_groups(folders, is_test_results=False):
    def get_event_file_name(folder_name):
        if not is_test_results:
            return [os.path.join(folder_name, "summaries", f) for f in os.listdir(os.path.join(folder_name, "summaries"))]
        else:
            all_files = []
            sub_folders = os.listdir(folder_name)
            for sub_folder in sub_folders:
                sub_path = os.path.join(folder_name, sub_folder)
                all_files += [os.path.join(sub_path, f) for f in os.listdir(sub_path)]
            return all_files

    groups = {}
    for folder_name in folders:
        # todo add check to separate per train dataset as well
        group_name = remove_run_suffix(folder_name=folder_name, is_test_results=is_test_results)

        if not group_name in groups.keys():
            groups[group_name] = []

        #todo add check to separate per test dataset
        groups[group_name] += get_event_file_name(folder_name)
    return groups


if __name__ == "__main__":
    # config = {
    #     "test_results_folder": "output/HumanoidImitation_30-16-51-12_test_results"
    # }
    # rp = ResultsPlotter(config)

    train_res_folders = ["output/HumanoidImitation_14-04-14-16-02", "output/HumanoidImitation_30-16-51-12",
                         "output/HumanoidImitationV2_14-14-22-41"]
    test_res_folders = [fol+"_test_results" for fol in train_res_folders]
    train_groups = create_groups(train_res_folders)
    test_groups = create_groups(test_res_folders, is_test_results=True)

    print(train_groups)
    print(test_groups)





"""
A1run1_test_results
    |--B1
    |--B2
A1run2_test_results
    |--B1
    |--B2
A2run1_test_results
    |--B1
    |--B2
    


_____Folder A averaging / vs:_____
Train results:
-average: over all run1, run2, ... with same A1
-vs.: A1 vs A2
Test results:
same

----Folder B averaging / vs:_____
Train results: None
Test results:
-average: all [todo] afterwards will be different
-vs.: TODO


[TODO] write in the train folder with which dataset trained
[TODO] write in the test folder with which dataset trained and with which dataset tested
Once done:
Train results:
    - average when same train_dataset
    - vs. different train_dataset
Test results:
    - average when same train_dataset and test datast
    - vs. different train_dataset or test dataset

"""