import itertools

import tensorboard as tb
import pandas as pd
from tensorboard.backend.event_processing import event_accumulator
import os
import re
import yaml

class ResultsPlotter:
    def __init__(self, groups):
        self.test_results_folder = config["test_results_folder"]
        event_acc = event_accumulator.EventAccumulator(os.path.join(self.test_results_folder, "_12-20-49-49"))
        event_acc.Reload()
        mnames = event_acc.Tags()
        print(mnames)

    def load_events(self, events_files_lists):
        return [event_accumulator.EventAccumulator(f) for f in events_files_lists]


def extract_group_parameters(results_path_list, is_test_results=False):
    """
    searches in the run config file for parameter that are used to group the results in different groups
    :return:
    """
    yaml.add_multi_constructor('tag:yaml.org,2002:python/object/new:isaacgym._bindings.linux-x86_64.gym_37.SimType',
                               lambda loader, suffix, node: None,
                               Loader=yaml.SafeLoader)
    yaml.add_multi_constructor('tag:yaml.org,2002:python/object:argparse.Namespace',
                               lambda loader, suffix, node: None,
                               Loader=yaml.SafeLoader)

    results_params = []
    for result_path in results_path_list:
        gr_params = {}
        # open run config file
        with open(os.path.join(result_path, "run_config.yaml"), 'r') as f:

            run_config = yaml.load(f, Loader=yaml.SafeLoader)

        #extracts tracking devices
        gr_params["track"] = run_config["cfg"]["env"]["asset"]["trackBodies"]

        #extracts algo name
        if is_test_results:
            # needs to load train config
            paths_parts = result_path.split(os.sep)
            train_result_path = os.path.join(paths_parts[0], paths_parts[1][:-(len("_test_results"))])
            with open(os.path.join(train_result_path, "run_config.yaml"), 'r') as f2:
                train_run_config = yaml.load(f2, Loader=yaml.SafeLoader)

            gr_params["algo"] = train_run_config["cfg_train"]["params"]["algo"]["name"]
        else:
            gr_params["algo"] = run_config["cfg_train"]["params"]["algo"]["name"]

        #extracts train dataset
        if is_test_results:
            gr_params["train_dataset"] = train_run_config["args"]["motion_file"]
        else:
            gr_params["train_dataset"] = run_config["args"]["motion_file"]

        #extracts if is real_time
        if is_test_results:
            gr_params["real_time"] = run_config["args"]["real_time"]
        else:
            gr_params["real_time"] = False

        #extracts test dataset
        if is_test_results:
            gr_params["test_dataset"] = run_config["args"]["motion_file"]
        else:
            gr_params["test_dataset"] = ""

        results_params.append(gr_params)

    return results_params, list(results_params[0].keys())


def create_groups(res_parameters, res_folders_paths, grouping_keys):
    to_group = None
    for i in range(len(grouping_keys)):
        keyfunc = lambda item: item[0][grouping_keys[i]]

        if to_group is None:
            to_group = [list(zip(res_parameters, res_folders_paths))]
        else:
            to_group = next_to_group

        next_to_group = []
        for data in to_group:
            data = sorted(data, key=keyfunc)
            for k, gr in itertools.groupby(data, keyfunc):
                #print(k)
                group = list(gr)
                #print(group)
                next_to_group.append(group)

    groups = []
    groups_parameters = []
    for group in next_to_group:
        groups_parameters.append(group[0][0])
        groups.append([g_el[1] for g_el in group])
    return groups, groups_parameters





if __name__ == "__main__":
    # config = {
    #     "test_results_folder": "output/HumanoidImitation_30-16-51-12_test_results"
    # }
    # rp = ResultsPlotter(config)



    # train_res_folders = ["output/HumanoidImitation_14-04-14-16-02", "output/HumanoidImitation_30-16-51-12",
    #                      "output/HumanoidImitationV2_14-14-22-41"]


    test_res_folders = [
     "output/HumanoidImitation_21-04-15-10-44_test_results/_21-04-16-45-27",
     "output/HumanoidImitation_21-04-15-10-44_test_results/_21-04-16-45-28",
     "output/HumanoidImitation_21-04-15-10-44_test_results/_21-04-16-45-29",
     "output/HumanoidImitation_21-04-15-10-44_test_results/_21-04-16-45-30",
     "output/HumanoidImitation_21-04-15-10-45_test_results/_21-04-16-45-27",
     "output/HumanoidImitation_21-04-15-10-45_test_results/_21-04-16-45-28",
     "output/HumanoidImitation_21-04-15-10-45_test_results/_21-04-16-45-29",
     "output/HumanoidImitation_21-04-15-10-45_test_results/_21-04-16-45-30"]



    test_res_parameters, grouping_keys = extract_group_parameters(test_res_folders, is_test_results=True)
    folders_groups, groups_parameters = create_groups(test_res_parameters, test_res_folders, grouping_keys)
    for j in range(len(folders_groups)):
        print(groups_parameters[j])
        print(folders_groups[j])

    #train_groups = create_groups(train_res_folders)
    #test_groups = create_groups(test_res_folders, is_test_results=True)

    #print(train_groups)
    #print(test_groups)





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