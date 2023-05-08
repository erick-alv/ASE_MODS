import itertools

import tensorboard as tb
import pandas as pd
from tensorboard.backend.event_processing import event_accumulator
import os
import re
import yaml


def extract_group_parameters(results_path_list, is_test_results=False, is_real_time=False):
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
            if is_real_time:
                train_result_path = os.path.join(paths_parts[0], paths_parts[1][:-(len("_test_results_real_time"))])
            else:
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


class GroupResults:
    def __init__(self, group_folders, is_test_results, is_real_time=False):
        self.is_test_results = is_test_results
        self.is_real_time = is_real_time
        self.load_events_into_dataframes(group_folders)
        if is_test_results and is_real_time:
            self.measures_plots_info = ["-aj_pos_error_track",
                                        "-aj_rot_error_track",
                                        "-pos_error_track",
                                        "-rot_error_track"]

            #creates average over steps and adds them to data_frame_dict
            d_av_over_steps = self.average_over_steps(self.measures_plots_info)
            self.data_frames_dict.update(d_av_over_steps)


            self.measures_single_vals = ["-as_aj_pos_error_track",
                                        "-as_aj_rot_error_track",
                                        "-as_pos_error_track",
                                        "-as_rot_error_track"]

        elif is_test_results:
            self.measures_plots_info = [
                ".npy-reward", ".npy-sip", ".npy-rot_error", ".npy-rot_error_track", ".npy-pos_error",
                ".npy-pos_error_track", ".npy-aj_jitter", ".npy-jitter"]

            self.measures_single_vals = [
                "am_reward", "am_weighted_reward", "am_as_aj_jitter",
                "am_as_aj_pos_error", "am_as_aj_pos_error_track",
                "am_as_aj_rot_error", "am_as_aj_rot_error_track",
                "am_as_sip",
                ".npy-cumulated_reward", ".npy-cumulated_weighted_reward", ".npy-as_sip",
                ".npy-as_aj_rot_error", ".npy-as_aj_rot_error_track", ".npy-as_aj_pos_error",
                ".npy-as_aj_pos_error_track", ".npy-as_rot_error", ".npy-as_rot_error_track", ".npy-as_pos_error",
                ".npy-as_pos_error_track", ".npy-as_aj_jitter"
            ]
        else:
            self.measures_plots_info = ["losses", "rewards0"]
            self.measures_single_vals = []

        self.plots_info, self.plots_info_std = self.average_over_runs(self.measures_plots_info)
        self.single_val_info, self.single_val_info_std = self.average_over_runs(self.measures_single_vals)

    def load_events_into_dataframes(self, events_files_lists):
        self.data_frames_dict = {}
        self.values_names = []
        for f in events_files_lists:
            if self.is_test_results:
                ev = event_accumulator.EventAccumulator(f)
            else:
                ev = event_accumulator.EventAccumulator(os.path.join(f, "summaries"))
            ev.Reload()
            mnames = ev.Tags()['scalars']
            for n in mnames:
                n_info = ev.Scalars(n)
                n_df = pd.DataFrame(n_info, columns=["wall_time", "step", n])
                n_df.drop("wall_time", axis=1, inplace=True)
                n_df = n_df.set_index("step")
                if not n in self.data_frames_dict.keys():
                    self.data_frames_dict[n] = []
                    self.values_names.append(n)
                self.data_frames_dict[n].append(n_df)

    def average_over_steps(self, measure_name_list):
        assert self.is_real_time
        df_dict_step_average = {}
        for name_measure in measure_name_list:
            #print("***************", name_measure, "***************")
            data_frames_k = [k for k in self.data_frames_dict.keys() if meas_mot_part(k).endswith(name_measure)]
            if name_measure.startswith(".npy-"):
                new_measure_name = ".npy-as_" + name_measure[5:]
            elif name_measure.startswith("-"):
                new_measure_name = "-as_" + name_measure[1:]
            else:
                new_measure_name = "as_" + name_measure

            for k in data_frames_k:
                new_k = k.replace(name_measure, new_measure_name)
                df_dict_step_average[new_k] = []
                #print("---", k, "---")
                for df in self.data_frames_dict[k]:
                    average = df.mean().item()
                    df_average = pd.DataFrame(data={"step": [0], new_k: [average]})
                    df_average = df_average.set_index("step")
                    df_dict_step_average[new_k].append(df_average)
                #print(df_dict_step_average[new_k])
        return df_dict_step_average

    def average_over_runs(self, measure_name_list):
        averaged_dict = {}
        std_dict = {}
        for name_measure in measure_name_list:
            #print("***************", name_measure, "***************")
            data_frames_k = [k for k in self.data_frames_dict.keys() if meas_mot_part(k).endswith(name_measure)]
            if name_measure.startswith(".npy-"):
                new_measure_name = ".npy-ar_"+name_measure[5:]
            elif name_measure.startswith("-"):
                new_measure_name = "-ar_" + name_measure[1:]
            else:
                new_measure_name = "ar_"+name_measure
            for k in data_frames_k:
                # print("---", k, "---")
                df_list = self.data_frames_dict[k]
                runs_concat = pd.concat(df_list)
                runs_grouped_step = runs_concat.reset_index().groupby("step")
                averaged_over_runs = runs_grouped_step.mean()
                std_over_runs = runs_grouped_step.std(ddof=0)
                new_k = k.replace(name_measure, new_measure_name) #todo why it becomes NaN
                averaged_dict[new_k] = averaged_over_runs
                std_dict[new_k] = std_over_runs
                # print(averaged_over_runs)
        return averaged_dict, std_dict


def meas_mot_part(measure_whole_name):
    return measure_whole_name.split("/")[0]

