from data_load import extract_group_parameters, create_groups, GroupResults
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os



colors = ["orange", "blue", "red", "green", "black", "purple", "gold", "cyan", "lime", "brown"]#


def smooth(df: pd.DataFrame, smooth_factor):
    #we use 1-smooth factor to have similar to tensorboard whn usinf smooth
    return df.ewm(alpha=(1-smooth_factor), adjust=True).mean()


def plot_groups_val(groups_res_list, groups_names, val_name, smooth_factor=None,
                    filename=None, do_plot=True,
                    title=None, xlabel=None, ylabel=None, legend_outside=False, plot_std=True):
    def plot_single(df_val, df_std, number_i):
        if smooth_factor is not None:
            df_val = smooth(df_val, smooth_factor)
            df_std = smooth(df_std, smooth_factor)

        x_axis_vals_array = df_val.index.values
        vals_array = df_val.iloc[:, 0].values
        std_array = df_std.iloc[:, 0].values

        label_name = groups_names[number_i]

        plt.plot(x_axis_vals_array, vals_array, label=label_name, color=colors[number_i])

        if plot_std:
            lower_bound = vals_array - std_array
            upper_bound = vals_array + std_array
            plt.fill_between(x_axis_vals_array, lower_bound, upper_bound, alpha=0.4, color=colors[number_i])

    sns.set_style("darkgrid")
    for i, gr_res in enumerate(groups_res_list):
        dfp_val = gr_res.plots_info[val_name]
        dfp_std = gr_res.plots_info_std[val_name]
        plot_single(dfp_val, dfp_std, number_i=i)

    if xlabel is not None:
        plt.xlabel(xlabel)
    if ylabel is not None:
        plt.ylabel(ylabel)
    if title is not None:
        plt.title(title)

    if legend_outside:
        plt.legend(bbox_to_anchor=(1.0, 1.0), loc='upper left')
        plt.tight_layout()
    else:
        plt.legend()


    if filename is not None:
        plt.savefig(filename)

    if do_plot:
        plt.show()


def create_plots_for_groups(groups_res_list, groups_attrs, groups_names, save_folder=None, is_test_results=False, is_real_time=False,
                            do_plot=True, extra_plot_title_name=None):
    if is_test_results:
        if is_real_time:
            val_names = ["real_time-ar_aj_pos_error_track", "real_time-ar_aj_rot_error_track",
                         'real_time-ar_pos_error_track/headset',  'real_time-ar_rot_error_track/headset']
            smooth_factor_list = [0.1, 0.1, 0.1, 0.1]
            to_title = {
                "real_time-ar_aj_pos_error_track": "MTPE",
                "real_time-ar_aj_rot_error_track": "MTRE",
                'real_time-ar_pos_error_track/headset': "MTPE (headset)",
                'real_time-ar_rot_error_track/headset': "MTRE (headset)"
            }
            to_ylabel = {
                "real_time-ar_aj_pos_error_track": "[cm]",
                "real_time-ar_aj_rot_error_track": "[deg]",
                'real_time-ar_pos_error_track/headset': "[cm]",
                'real_time-ar_rot_error_track/headset': "[deg]"
            }
            to_xlabel = {}
            #every x_label is step
            for k in val_names:
                to_xlabel[k] = "step"
        else:
            # list for the motions that we want to plot the reward
            # E.g.: val_names = ['07_01_cmu_amp.npy-ar_reward']
            val_names = []
            smooth_factor_list = [0.2]
            # dictionary with the names we want as title
            # E.g.:
            # to_title = {
            #     '07_01_cmu_amp.npy-ar_reward': 'Reward 07_01_cmu_amp',
            # }
            to_title = {}

            to_ylabel = {}
            to_xlabel = {}
            # every x_label is step and y label reward
            for k in val_names:
                to_ylabel[k] = ""
                to_xlabel[k] = "step"

    else:
        val_names = ['ar_losses/a_loss', 'ar_losses/c_loss', 'ar_losses/entropy', 'ar_rewards0/frame']
        smooth_factor_list = [0.9, 0.9, 0.9, 0.9]
        to_title = {
            'ar_losses/a_loss': "Actor loss",
            'ar_losses/c_loss': "Critic loss",
            'ar_losses/entropy': "Entropy",
            'ar_rewards0/frame': "Reward"
        }
        to_ylabel = {}
        to_xlabel = {}
        # every x_label is step and y label reward
        for k in val_names:
            to_ylabel[k] = ""
            to_xlabel[k] = ""


    for i, val_name in enumerate(val_names):
        smooth_factor = smooth_factor_list[i]
        title = to_title[val_name]
        if extra_plot_title_name is not None:
            title = title + " " + extra_plot_title_name
        xlabel = to_xlabel[val_name]
        ylabel = to_ylabel[val_name]
        if save_folder is not None:
            endname = title.replace(" ", "_") + ".png"
            filename = os.path.join(save_folder, endname)
        else:
            filename = None

        plot_groups_val(groups_res_list, groups_names, val_name=val_name, smooth_factor=smooth_factor,
                        title=title, xlabel=xlabel, ylabel=ylabel,
                        legend_outside=False, plot_std=True, do_plot=do_plot,
                        filename=filename)

def create_tables_for_groups(groups_res_list, groups_attrs, groups_names, save_folder=None, is_test_results=False,
                             is_real_time=False, do_print=False):

    def fill_table_dict(table_dict):
        for i, gr_res in enumerate(groups_res_list):
            for key in table_dict.keys():
                if key == "method":
                    group_name = groups_names[i]
                    table_dict[key].append(group_name)
                else:
                    val = gr_res.single_val_info[key].iloc[0].item()
                    table_dict[key].append(val)
        return table_dict

    def switch_key_names(table_dict, name_dict):
        s_dict = {}
        for key in table_dict.keys():
            if (key in name_dict.keys()):
                s_dict[name_dict[key]] = table_dict[key]
            else:
                s_dict[key] = table_dict[key]
        return s_dict

    def create_table_dataframe(table_dict):
        table_df = pd.DataFrame(table_dict)
        table_df = table_df.set_index("method")
        table_df = table_df.transpose()
        return table_df

    def save_dataframe(df, filename):
        s_folder = os.path.dirname(filename)
        if not os.path.exists(s_folder):
            os.makedirs(s_folder)
        with open(filename, "w") as f:
            f.write(df.to_latex(float_format="{:.4f}".format))

    if is_test_results and is_real_time:
        errors_table_dict = {
            "method": [],
            'real_time-ar_as_aj_pos_error_track': [],
            'real_time-ar_as_aj_rot_error_track': [],
            'real_time-ar_as_pos_error_track/headset': [],
            'real_time-ar_as_pos_error_track/right_controller': [],
            'real_time-ar_as_pos_error_track/left_controller': [],
            'real_time-ar_as_rot_error_track/headset': [],
            'real_time-ar_as_rot_error_track/right_controller': [],
            'real_time-ar_as_rot_error_track/left_controller': []

        }
        sim_name_to_table_name = {
            'real_time-ar_as_aj_pos_error_track': 'MTPE [cm]',
            'real_time-ar_as_aj_rot_error_track': 'MTRE [deg]',
            'real_time-ar_as_pos_error_track/headset': 'MTPE [cm] (headset)',
            'real_time-ar_as_pos_error_track/right_controller': 'MTPE [cm] (right controller)',
            'real_time-ar_as_pos_error_track/left_controller': 'MTPE [cm] (left controller)',
            'real_time-ar_as_rot_error_track/headset': 'MTRE [deg] (headset)',
            'real_time-ar_as_rot_error_track/right_controller': 'MTRE [deg] (right controller)',
            'real_time-ar_as_rot_error_track/left_controller': 'MTRE [deg] (left controller)'
        }
        errors_table_dict = fill_table_dict(errors_table_dict)
        errors_table_dict = switch_key_names(errors_table_dict, sim_name_to_table_name)
        errors_df = create_table_dataframe(errors_table_dict)

        if save_folder is not None:
            save_dataframe(errors_df, os.path.join(save_folder, "real_time_errors_table.txt"))

        if do_print:
            print(errors_df.to_latex(float_format="{:.2f}".format))

    elif is_test_results:
        errors_table_dict = {
            "method": [],
            'ar_am_as_aj_jitter': [],
            'ar_am_as_aj_jitter_gt': [],
            'ar_am_as_aj_pos_error': [],
            'ar_am_as_aj_pos_error_track': [],
            'ar_am_as_aj_rot_error': [],
            'ar_am_as_aj_rot_error_track': [],
            'ar_am_as_sip': [],
            'ar_am_as_loc_error': []

        }
        rewards_table_dict = {
            "method": [],
            'ar_am_cumulated_reward': [],
            'ar_am_cumulated_weighted_reward': []
        }
        sim_name_to_table_name = {
            'ar_am_as_aj_jitter': 'Jitter [km/s³]',
            'ar_am_as_aj_jitter_gt': 'Jitter GT [km/s³]',
            'ar_am_as_aj_pos_error': 'MPJPE [cm]',
            'ar_am_as_aj_pos_error_track': 'MTPE [cm]',
            'ar_am_as_loc_error': 'MLE [cm]',
            'ar_am_as_aj_rot_error': 'MPJRE [deg]',
            'ar_am_as_aj_rot_error_track': 'MTRE [deg]',
            'ar_am_as_sip': 'SIP [deg]',
            'ar_am_cumulated_reward': 'reward',
            'ar_am_cumulated_weighted_reward': 'weighted reward'
        }

        errors_table_dict = fill_table_dict(errors_table_dict)
        errors_table_dict = switch_key_names(errors_table_dict, sim_name_to_table_name)
        errors_df = create_table_dataframe(errors_table_dict)

        rewards_table_dict = fill_table_dict(rewards_table_dict)
        rewards_table_dict = switch_key_names(rewards_table_dict, sim_name_to_table_name)
        rewards_df = create_table_dataframe(rewards_table_dict)

        if save_folder is not None:

            save_dataframe(errors_df, os.path.join(save_folder, "errors_table.txt"))
            save_dataframe(rewards_df, os.path.join(save_folder, "rewards_table.txt"))

        if do_print:
            print(errors_df.to_latex(float_format="{:.2f}".format))
            print(rewards_df.to_latex(float_format="{:.2f}".format))
    else:
        #no single rewards here
        pass


#todo code to transform group attr to a name for display (preferably method that transform element by element than
# single dict written manually)
#todo code to transform the names of datasets

def get_groups_names(groups_parameters, grouping_keys_to_use, val_to_name_dict):
    names_groups = [
        get_name_from_parameters(parameter_dict, grouping_keys_to_use, val_to_name_dict) for parameter_dict in groups_parameters]
    return names_groups

def get_name_from_parameters(parameter_dict, grouping_keys_to_use, val_to_name_dict):
    element_name = ""
    for i in range(len(grouping_keys_to_use)):
        g_key = grouping_keys_to_use[i]
        parameter_val = parameter_dict[g_key]
        val_name = val_to_name_dict[g_key][parameter_val]
        if i > 0:
            element_name += "-"
        element_name += val_name
    return element_name

if __name__ == "__main__":

    #to save just specify in which folder; then function decides which name to give

    res = 6
    if res == 1:

        # runs results
        runs_res_folders = [
            "output_temp/HumanoidImitation_21-04-15-10-44",
            "output_temp/HumanoidImitation_21-04-15-10-45"
        ]
        # runs_res_folders = [
        #     "output2/HumanoidImitation_29-04-09-09-50",
        #     "output2/HumanoidImitation_03-05-10-59-29"
        # ]

        runs_res_parameters, grouping_keys = extract_group_parameters(runs_res_folders, is_test_results=False)
        folders_groups, groups_parameters = create_groups(runs_res_parameters, runs_res_folders, grouping_keys)
        for fols in folders_groups:
            print(fols)
        gr_res_list = [GroupResults(fols, is_test_results=False, is_real_time=False) for fols in
                       folders_groups]

        print(list(gr_res_list[0].plots_info.keys()))

        #create_plots_for_groups(gr_res_list, groups_parameters)
        create_tables_for_groups(gr_res_list, groups_parameters)

    elif res == 2:

        #  test results
        test_res_folders = [
         "output_temp/HumanoidImitation_21-04-15-10-44_test_results/_21-04-16-45-27",
         "output_temp/HumanoidImitation_21-04-15-10-44_test_results/_21-04-16-45-28",
         "output_temp/HumanoidImitation_21-04-15-10-44_test_results/_21-04-16-45-29",
         "output_temp/HumanoidImitation_21-04-15-10-44_test_results/_21-04-16-45-30",
         "output_temp/HumanoidImitation_21-04-15-10-45_test_results/_21-04-16-45-27",
         "output_temp/HumanoidImitation_21-04-15-10-45_test_results/_21-04-16-45-28",
         "output_temp/HumanoidImitation_21-04-15-10-45_test_results/_21-04-16-45-29",
         "output_temp/HumanoidImitation_21-04-15-10-45_test_results/_21-04-16-45-30"]

        test_res_parameters, grouping_keys = extract_group_parameters(test_res_folders, is_test_results=True)
        folders_groups, groups_parameters = create_groups(test_res_parameters, test_res_folders, grouping_keys)
        for fols in folders_groups:
            print(fols)
        gr_res_list = [GroupResults(fols, is_test_results=True, is_real_time=False) for fols in
                       folders_groups]



        create_plots_for_groups(gr_res_list, groups_parameters, is_test_results=True, is_real_time=False,
                                save_folder='plots_temp')
        create_tables_for_groups(gr_res_list, groups_parameters, is_test_results=True, is_real_time=False,
                                 do_print=True, save_folder="tables_temp")

    elif res == 3:
        #  test results real time
        real_time_res_folders = [
            "output_temp/HumanoidImitation_21-04-15-10-44_test_results_real_time/_03-05-14-11-19"
        ]
        real_time_res_parameters, grouping_keys = extract_group_parameters(real_time_res_folders, is_test_results=True,
                                                                           is_real_time=True)
        folders_groups, groups_parameters = create_groups(real_time_res_parameters, real_time_res_folders, grouping_keys)
        for fol in folders_groups:
            print(fol)
        gr_res_list = [GroupResults(fols, is_test_results=True, is_real_time=True) for fols in
                       folders_groups]

        print(list(gr_res_list[0].plots_info.keys()))

        #create_plots_for_groups(gr_res_list, groups_parameters, is_test_results=True, is_real_time=True)
        create_tables_for_groups(gr_res_list, groups_parameters, is_test_results=True, is_real_time=True,
                                 do_print=True, save_folder="tables_temp")
    elif res == 4: #For tests of different versions of other setup.
        folder_res = [
            "output_Deb_otherSetup_test_res/HumanoidImitation_14-06-19-56-54_test_results/_22-06-10-36-25",
            "output_Deb_otherSetup_test_res/HumanoidImitation_16-06-21-43-43_test_results/_22-06-10-35-43",
            "output_Deb_otherSetup_test_res/HumanoidImitation_17-06-18-03-25_test_results/_22-06-10-37-14",
            "output_Deb_otherSetup_test_res/HumanoidImitation_17-06-18-09-12_test_results/_22-06-10-37-56"
        ]

        # grouping_keys_to_use = ['cfg_train', 'cfg_env_train', 'cfg_env_test', 'train_dataset', 'test_dataset',
        #                         'checkpoint']
        grouping_keys_to_use = ['cfg_env_train']
        val_to_name_dict = {
            "cfg_env_train": {
                "ase/data/cfg/other_setup/humanoid_imitation_vrh_pd_rewPenaltyAndReach_v01.yaml": "v01",
                "ase/data/cfg/other_setup/humanoid_imitation_vrh_pd_rewPenaltyAndReach_v02.yaml": "v02",
                "ase/data/cfg/other_setup/humanoid_imitation_vrh_pd_rewPenaltyAndReach_v03.yaml": "v03",
                "ase/data/cfg/other_setup/humanoid_imitation_vrh_pd_rewPenaltyAndReach_v04.yaml": "v04"
            }
        }

        parameters, grouping_keys = extract_group_parameters(folder_res, is_test_results=True, is_real_time=False)
        for k in grouping_keys_to_use:
            assert k in grouping_keys
        folders_groups, groups_parameters = create_groups(parameters, folder_res, grouping_keys_to_use)
        names_groups = get_groups_names(groups_parameters, grouping_keys_to_use, val_to_name_dict)

        gr_res_list = [GroupResults(fols, is_test_results=True, is_real_time=False) for fols in
                       folders_groups]

        for gn in names_groups:
            print(gn)

        create_tables_for_groups(gr_res_list, groups_parameters, names_groups, is_test_results=True, is_real_time=False,
                                 do_print=False, save_folder="output_Deb_otherSetup_test_res")

    elif res == 5: #For tests of different times
        folder_res = [
            "output_final_exps_fullvshalf/HumanoidImitation_18-06-05-01-55_test_results/_23-06-01-16-28",
            "output_final_exps_fullvshalf/HumanoidImitation_15-06-18-49-50_test_results/_23-06-00-58-22",
            "output_final_exps_fullvshalf/HumanoidImitation_15-06-18-49-50_test_results/_23-06-00-40-09",
            "output_final_exps_fullvshalf/HumanoidImitation_15-06-18-49-50_test_results/_22-06-11-36-40",
            "output_final_exps_fullvshalf/HumanoidImitation_15-06-18-49-50_test_results/_22-06-11-18-06",
            "output_final_exps_fullvshalf/HumanoidImitation_14-06-16-45-17_test_results/_23-06-02-10-05",
            "output_final_exps_fullvshalf/HumanoidImitation_14-06-16-45-17_test_results/_23-06-01-52-10",
            "output_final_exps_fullvshalf/HumanoidImitation_14-06-16-45-17_test_results/_23-06-01-34-18",
            "output_final_exps_fullvshalf/HumanoidImitation_11-06-12-25-55_test_results/_23-06-00-22-38",
            "output_final_exps_fullvshalf/HumanoidImitation_11-06-12-25-55_test_results/_23-06-00-03-33",
            "output_final_exps_fullvshalf/HumanoidImitation_11-06-12-25-55_test_results/_22-06-12-37-56",
            "output_final_exps_fullvshalf/HumanoidImitation_11-06-12-25-55_test_results/_22-06-12-14-48",
            "output_final_exps_fullvshalf/HumanoidImitation_11-06-12-25-55_test_results/_22-06-11-56-51",
            "output_final_exps_fullvshalf/HumanoidImitation_19-06-01-53-04_test_results/_23-06-12-34-08",
            "output_final_exps_fullvshalf/HumanoidImitation_19-06-01-53-04_test_results/_23-06-12-52-18",
            "output_final_exps_fullvshalf/HumanoidImitation_19-06-01-53-04_test_results/_23-06-13-10-18",
            "output_final_exps_fullvshalf/HumanoidImitation_19-06-01-53-04_test_results/_23-06-16-32-49"
        ]

        # grouping_keys_to_use = ['cfg_train', 'cfg_env_train', 'cfg_env_test', 'train_dataset', 'test_dataset',
        #                         'checkpoint']
        grouping_keys_to_use = ['cfg_env_train', 'test_dataset', 'cfg_train', 'checkpoint']
        val_to_name_dict = {
            "cfg_env_train": {
                "ase/data/cfg/humanoid_imitation_vrh.yaml": "rewQuestSim",
                "ase/data/cfg/humanoid_imitation_vrh_rewPenaltyAndReach.yaml": "rewPenaltyAndReach",
            },
            'cfg_train':{
                "ase/data/cfg/train/rlg/common_ppo_humanoid_v3.yaml": "trainv3",
                "ase/data/cfg/train/rlg/common_ppo_humanoid.yaml": "trainvNormal"

            },
            'test_dataset': {
                "ase/data/motions/dataset_questsim_test.yaml": "DATASETTEST",
                "ase/data/motions/dataset_lafan_test.yaml": "LAFAN"

            },
            "checkpoint": {
                "output_final_exps/HumanoidImitation_11-06-12-25-55/nn/HumanoidImitation.pth":"full",
                "output_final_exps/HumanoidImitation_11-06-12-25-55/nn/HumanoidImitation_000100000.pth":"100",
                "output_final_exps/HumanoidImitation_11-06-12-25-55/nn/HumanoidImitation_000150000.pth": "150",
                "output_final_exps/HumanoidImitation_11-06-12-25-55/nn/HumanoidImitation_000050000.pth": "050",
                "output_final_exps/HumanoidImitation_15-06-18-49-50/nn/HumanoidImitation.pth": "full",
                "output_final_exps/HumanoidImitation_15-06-18-49-50/nn/HumanoidImitation_000100000.pth": "100",
                "output_final_exps/HumanoidImitation_15-06-18-49-50/nn/HumanoidImitation_000150000.pth": "150",
                "output_final_exps/HumanoidImitation_15-06-18-49-50/nn/HumanoidImitation_000050000.pth": "050",
                "output_final_exps/HumanoidImitation_19-06-01-53-04/nn/HumanoidImitation.pth": "full",
                "output_final_exps/HumanoidImitation_19-06-01-53-04/nn/HumanoidImitation_000100000.pth": "100",
                "output_final_exps/HumanoidImitation_19-06-01-53-04/nn/HumanoidImitation_000150000.pth": "150",
                "output_final_exps/HumanoidImitation_19-06-01-53-04/nn/HumanoidImitation_000050000.pth": "050",
                "output_final_exps/HumanoidImitation_18-06-05-01-55/nn/HumanoidImitation.pth": "full",
                "output_final_exps/HumanoidImitation_14-06-16-45-17/nn/HumanoidImitation_000100000.pth": "100",
                "output_final_exps/HumanoidImitation_14-06-16-45-17/nn/HumanoidImitation_000150000.pth": "150",
                "output_final_exps/HumanoidImitation_14-06-16-45-17/nn/HumanoidImitation_000050000.pth": "050",
            }
        }

        parameters, grouping_keys = extract_group_parameters(folder_res, is_test_results=True, is_real_time=False)
        for k in grouping_keys_to_use:
            assert k in grouping_keys
        folders_groups, groups_parameters = create_groups(parameters, folder_res, grouping_keys_to_use)
        names_groups = get_groups_names(groups_parameters, grouping_keys_to_use, val_to_name_dict)

        gr_res_list = [GroupResults(fols, is_test_results=True, is_real_time=False) for fols in
                       folders_groups]

        for gn in names_groups:
            print(gn)

        create_tables_for_groups(gr_res_list, groups_parameters, names_groups, is_test_results=True, is_real_time=False,
                                 do_print=False, save_folder="output_final_exps_fullvshalf")

    elif res == 6:
        sub_res = 16

        if sub_res == 0:
            save_folder_name = "output_final_exps_test_results"
            folder_res = [
                "output_final_exps_test_results/HumanoidImitation_11-06-12-25-55_test_results/_24-06-12-58-48",
                "output_final_exps_test_results/HumanoidImitation_11-06-12-25-55_test_results/_24-06-13-16-54",
                "output_final_exps_test_results/HumanoidImitation_14-06-16-45-17_test_results/_24-06-16-47-39",
                "output_final_exps_test_results/HumanoidImitation_14-06-16-45-17_test_results/_24-06-17-07-01",
                "output_final_exps_test_results/HumanoidImitation_15-06-18-49-50_test_results/_24-06-13-36-09",
                "output_final_exps_test_results/HumanoidImitation_15-06-18-49-50_test_results/_24-06-13-55-25",
                "output_final_exps_test_results/HumanoidImitation_19-06-01-53-04_test_results/_24-06-16-08-00",
                "output_final_exps_test_results/HumanoidImitation_19-06-01-53-04_test_results/_24-06-16-27-35",
                "output_final_exps_test_results/HumanoidImitation_20-06-06-19-02_test_results/_24-06-17-26-40",
                "output_final_exps_test_results/HumanoidImitation_20-06-06-19-02_test_results/_24-06-17-38-44",
                "output_final_exps_test_results/HumanoidImitation_20-06-16-24-51_test_results/_24-06-14-14-19",
                "output_final_exps_test_results/HumanoidImitation_20-06-16-24-51_test_results/_24-06-15-56-15"
            ]
        elif sub_res == 1:
            save_folder_name = "output_final_exps_test_results/HumanoidImitation_11-06-12-25-55_test_results"
            folder_res = [
                "output_final_exps_test_results/HumanoidImitation_11-06-12-25-55_test_results/_24-06-12-58-48",
                "output_final_exps_test_results/HumanoidImitation_11-06-12-25-55_test_results/_24-06-13-16-54",
                "output_final_exps_test_results/HumanoidImitation_11-06-12-25-55_test_results/_24-06-23-39-55",
                "output_final_exps_test_results/HumanoidImitation_11-06-12-25-55_test_results/_25-06-01-49-32"

            ]
        elif sub_res == 2:
            save_folder_name = "output_final_exps_test_results/HumanoidImitation_14-06-16-45-17_test_results"
            folder_res = [
                "output_final_exps_test_results/HumanoidImitation_14-06-16-45-17_test_results/_24-06-16-47-39",
                "output_final_exps_test_results/HumanoidImitation_14-06-16-45-17_test_results/_24-06-17-07-01",
                "output_final_exps_test_results/HumanoidImitation_14-06-16-45-17_test_results/_25-06-18-52-41",
                "output_final_exps_test_results/HumanoidImitation_14-06-16-45-17_test_results/_26-06-02-29-57"
            ]
        elif sub_res == 3:
            save_folder_name = "output_final_exps_test_results/HumanoidImitation_15-06-18-49-50_test_results"
            folder_res = [
                "output_final_exps_test_results/HumanoidImitation_15-06-18-49-50_test_results/_24-06-13-36-09",
                "output_final_exps_test_results/HumanoidImitation_15-06-18-49-50_test_results/_24-06-13-55-25",
                "output_final_exps_test_results/HumanoidImitation_15-06-18-49-50_test_results/_25-06-03-56-23",
                "output_final_exps_test_results/HumanoidImitation_15-06-18-49-50_test_results/_25-06-06-06-42"
            ]
        elif sub_res == 4:
            save_folder_name = "output_final_exps_test_results/HumanoidImitation_19-06-01-53-04_test_results"
            folder_res = [
                "output_final_exps_test_results/HumanoidImitation_19-06-01-53-04_test_results/_24-06-16-08-00",
                "output_final_exps_test_results/HumanoidImitation_19-06-01-53-04_test_results/_24-06-16-27-35",
                "output_final_exps_test_results/HumanoidImitation_19-06-01-53-04_test_results/_25-06-22-09-54",
                "output_final_exps_test_results/HumanoidImitation_19-06-01-53-04_test_results/_26-06-00-20-36"
                ]
        elif sub_res == 5:
            save_folder_name = "output_final_exps_test_results/HumanoidImitation_20-06-06-19-02_test_results"
            folder_res = [
                "output_final_exps_test_results/HumanoidImitation_20-06-06-19-02_test_results/_24-06-17-26-40",
                "output_final_exps_test_results/HumanoidImitation_20-06-06-19-02_test_results/_24-06-17-38-44",
                "output_final_exps_test_results/HumanoidImitation_20-06-06-19-02_test_results/_25-06-15-46-02",
                "output_final_exps_test_results/HumanoidImitation_20-06-06-19-02_test_results/_25-06-17-13-43"
            ]
        elif sub_res == 6:
            save_folder_name = "output_final_exps_test_results/HumanoidImitation_20-06-16-24-51_test_results"
            folder_res = [
                "output_final_exps_test_results/HumanoidImitation_20-06-16-24-51_test_results/_24-06-14-14-19",
                "output_final_exps_test_results/HumanoidImitation_20-06-16-24-51_test_results/_24-06-15-56-15",
                "output_final_exps_test_results/HumanoidImitation_20-06-16-24-51_test_results/_25-06-08-52-01",
                "output_final_exps_test_results/HumanoidImitation_20-06-16-24-51_test_results/_25-06-10-17-56"
            ]
        ######################## For moresimilar
        elif sub_res == 7:
            save_folder_name = "output_final_exps_DifferentParameters_moresimilar/HumanoidImitation_11-06-12-25-55_test_results"
            folder_res = [
                "output_final_exps_DifferentParameters_moresimilar/HumanoidImitation_11-06-12-25-55_test_results/_03-07-23-00-49",
                "output_final_exps_DifferentParameters_moresimilar/HumanoidImitation_11-06-12-25-55_test_results/_03-07-23-36-53",
                "output_final_exps_DifferentParameters_moresimilar/HumanoidImitation_11-06-12-25-55_test_results/_04-07-00-16-38",
"output_final_exps_DifferentParameters_moresimilar/HumanoidImitation_11-06-12-25-55_test_results/_04-07-00-12-44"
            ]
        elif sub_res == 8:
            save_folder_name = "output_final_exps_DifferentParameters_moresimilar/HumanoidImitation_14-06-16-45-17_test_results"
            folder_res = [
                "output_final_exps_DifferentParameters_moresimilar/HumanoidImitation_14-06-16-45-17_test_results/_04-07-09-42-33",
                "output_final_exps_DifferentParameters_moresimilar/HumanoidImitation_14-06-16-45-17_test_results/_04-07-10-21-23",
                "output_final_exps_DifferentParameters_moresimilar/HumanoidImitation_14-06-16-45-17_test_results/_04-07-12-16-17",
"output_final_exps_DifferentParameters_moresimilar/HumanoidImitation_14-06-16-45-17_test_results/_04-07-12-20-27"
            ]
        elif sub_res == 9:
            save_folder_name = "output_final_exps_DifferentParameters_moresimilar/HumanoidImitation_15-06-18-49-50_test_results"
            folder_res = [
                "output_final_exps_DifferentParameters_moresimilar/HumanoidImitation_15-06-18-49-50_test_results/_04-07-01-42-49",
"output_final_exps_DifferentParameters_moresimilar/HumanoidImitation_15-06-18-49-50_test_results/_04-07-02-20-36",
"output_final_exps_DifferentParameters_moresimilar/HumanoidImitation_15-06-18-49-50_test_results/_04-07-02-57-58",
"output_final_exps_DifferentParameters_moresimilar/HumanoidImitation_15-06-18-49-50_test_results/_04-07-03-01-54"
            ]
        elif sub_res == 10:
            save_folder_name = "output_final_exps_DifferentParameters_moresimilar/HumanoidImitation_19-06-01-53-04_test_results"
            folder_res = [
                "output_final_exps_DifferentParameters_moresimilar/HumanoidImitation_19-06-01-53-04_test_results/_04-07-00-20-29",
"output_final_exps_DifferentParameters_moresimilar/HumanoidImitation_19-06-01-53-04_test_results/_04-07-00-57-30",
"output_final_exps_DifferentParameters_moresimilar/HumanoidImitation_19-06-01-53-04_test_results/_04-07-01-34-52",
"output_final_exps_DifferentParameters_moresimilar/HumanoidImitation_19-06-01-53-04_test_results/_04-07-01-38-48"
            ]
        elif sub_res == 11:
            save_folder_name = "output_final_exps_DifferentParameters_moresimilar/HumanoidImitation_20-06-06-19-02_test_results"
            folder_res = [
                "output_final_exps_DifferentParameters_moresimilar/HumanoidImitation_20-06-06-19-02_test_results/_04-07-12-26-16",
"output_final_exps_DifferentParameters_moresimilar/HumanoidImitation_20-06-06-19-02_test_results/_04-07-12-51-06",
"output_final_exps_DifferentParameters_moresimilar/HumanoidImitation_20-06-06-19-02_test_results/_04-07-13-16-08",
"output_final_exps_DifferentParameters_moresimilar/HumanoidImitation_20-06-06-19-02_test_results/_04-07-13-18-48"
            ]
        elif sub_res == 12:
            save_folder_name = "output_final_exps_DifferentParameters_moresimilar/HumanoidImitation_20-06-16-24-51_test_results"
            folder_res = [
                "output_final_exps_DifferentParameters_moresimilar/HumanoidImitation_20-06-16-24-51_test_results/_04-07-03-05-56",
"output_final_exps_DifferentParameters_moresimilar/HumanoidImitation_20-06-16-24-51_test_results/_04-07-03-29-35",
"output_final_exps_DifferentParameters_moresimilar/HumanoidImitation_20-06-16-24-51_test_results/_04-07-03-53-19",
"output_final_exps_DifferentParameters_moresimilar/HumanoidImitation_20-06-16-24-51_test_results/_04-07-03-55-52"
            ]
        elif sub_res == 13:
            save_folder_name = "output_final_exps_1_72_test_results/HumanoidImitation_02-07-20-53-34_test_results"
            folder_res = [
                "output_final_exps_1_72_test_results/HumanoidImitation_02-07-20-53-34_test_results/_11-07-23-54-09",
                "output_final_exps_1_72_test_results/HumanoidImitation_02-07-20-53-34_test_results/_12-07-00-20-32",
                "output_final_exps_1_72_test_results/HumanoidImitation_02-07-20-53-34_test_results/_13-07-04-17-25",
                "output_final_exps_1_72_test_results/HumanoidImitation_02-07-20-53-34_test_results/_15-07-05-03-02"
            ]
        elif sub_res == 14:
            save_folder_name = "output_final_exps_1_72_test_results/HumanoidImitation_03-07-06-17-52_test_results"
            folder_res = [
                "output_final_exps_1_72_test_results/HumanoidImitation_03-07-06-17-52_test_results/_11-07-22-25-58",
                "output_final_exps_1_72_test_results/HumanoidImitation_03-07-06-17-52_test_results/_11-07-23-10-13",
                "output_final_exps_1_72_test_results/HumanoidImitation_03-07-06-17-52_test_results/_13-07-01-17-19",
                "output_final_exps_1_72_test_results/HumanoidImitation_03-07-06-17-52_test_results/_13-07-02-47-24"
            ]
        elif sub_res == 15:
            save_folder_name = "output_final_exps_1_72_test_results/HumanoidImitation_07-07-18-24-42_test_results"
            folder_res = [
                "output_final_exps_1_72_test_results/HumanoidImitation_07-07-18-24-42_test_results/_12-07-23-47-44",
                "output_final_exps_1_72_test_results/HumanoidImitation_07-07-18-24-42_test_results/_13-07-00-32-18",
                "output_final_exps_1_72_test_results/HumanoidImitation_07-07-18-24-42_test_results/_15-07-02-02-42",
                "output_final_exps_1_72_test_results/HumanoidImitation_07-07-18-24-42_test_results/_15-07-03-32-01"
            ]
        elif sub_res == 16:
            save_folder_name = "output_final_exps_1_72_test_results/HumanoidImitation_09-07-08-30-33_test_results"
            folder_res = [
                "output_final_exps_1_72_test_results/HumanoidImitation_09-07-08-30-33_test_results/_12-07-22-17-13",
                "output_final_exps_1_72_test_results/HumanoidImitation_09-07-08-30-33_test_results/_12-07-23-02-03",
                "output_final_exps_1_72_test_results/HumanoidImitation_09-07-08-30-33_test_results/_13-07-22-33-44",
                "output_final_exps_1_72_test_results/HumanoidImitation_09-07-08-30-33_test_results/_14-07-23-02-22",
                "output_final_exps_1_72_test_results/HumanoidImitation_09-07-08-30-33_test_results/_15-07-00-32-26"
            ]

        # grouping_keys_to_use = ['cfg_train', 'cfg_env_train', 'cfg_env_test', 'train_dataset', 'test_dataset',
        #                         'checkpoint']
        grouping_keys_to_use = ['cfg_env_train', 'cfg_env_test', 'cfg_train', 'test_dataset']
        val_to_name_dict = {
            'cfg_train': {
                "ase/data/cfg/train/rlg/common_ppo_humanoid_v3.yaml": "trainv3",
                "ase/data/cfg/train/rlg/common_ppo_humanoid.yaml": "trainvNormal"
            },
            "cfg_env_train": {
                "ase/data/cfg/humanoid_imitation_vrh.yaml": "(1/36)rewQuestSimTorque",
                "ase/data/cfg/humanoid_imitation_vrh_rewPenaltyAndReach.yaml": "(1/36)rewPenaltyAndReachTorque",
                "ase/data/cfg/humanoid_imitation_vrh_pd.yaml": "(1/36)rewQuestSimPd",
                "ase/data/cfg/humanoid_imitation_vrh_pd_rewPenaltyAndReach.yaml": "(1/36)rewPenaltyAndReachPd",
                "ase/data/cfg/other_setup/humanoid_imitation_vrh.yaml": "(1/72)rewQuestSimTorque",
                "ase/data/cfg/other_setup/humanoid_imitation_vrh_rewPenaltyAndReach.yaml": "(1/72)rewPenaltyAndReachTorque",
                "ase/data/cfg/other_setup/humanoid_imitation_vrh_pd.yaml": "(1/72)rewQuestSimPd",
                "ase/data/cfg/other_setup/humanoid_imitation_vrh_pd_rewPenaltyAndReach.yaml": "(1/72)rewPenaltyAndReachPd"
            },
            "cfg_env_test": {
                "ase/data/cfg/humanoid_imitation_vrh.yaml": "TestRewQuestSim",
                "ase/data/cfg/humanoid_imitation_vrh_rewPenaltyAndReach.yaml": "TestRewPenaltyAndReach",
                "ase/data/cfg/humanoid_imitation_vrh_pd.yaml": "TestRewQuestSim",
                "ase/data/cfg/humanoid_imitation_vrh_pd_rewPenaltyAndReach.yaml":"TestRewPenaltyAndReach",
                "ase/data/cfg/other_setup/humanoid_imitation_vrh.yaml": "TestRewQuestSim",
                "ase/data/cfg/other_setup/humanoid_imitation_vrh_rewPenaltyAndReach.yaml": "TestRewPenaltyAndReach",
                "ase/data/cfg/other_setup/humanoid_imitation_vrh_pd.yaml": "TestRewQuestSim",
                "ase/data/cfg/other_setup/humanoid_imitation_vrh_pd_rewPenaltyAndReach.yaml": "TestRewPenaltyAndReach"
            },
            'test_dataset': {
                "ase/data/motions/dataset_questsim_test.yaml": "DATASETTEST",
                "ase/data/motions/dataset_lafan_test.yaml": "LAFAN",
                "ase/data/motions/dataset_lafanlocomotion_test.yaml": "LAFANSIMILAR",
                "ase/data/motions/dataset_lafansomewhat_test.yaml": "LAFANSOME",


            }
        }

        parameters, grouping_keys = extract_group_parameters(folder_res, is_test_results=True, is_real_time=False)
        for k in grouping_keys_to_use:
            assert k in grouping_keys
        folders_groups, groups_parameters = create_groups(parameters, folder_res, grouping_keys_to_use)
        names_groups = get_groups_names(groups_parameters, grouping_keys_to_use, val_to_name_dict)

        gr_res_list = [GroupResults(fols, is_test_results=True, is_real_time=False) for fols in
                       folders_groups]

        for gn in names_groups:
            print(gn)



        create_tables_for_groups(gr_res_list, groups_parameters, names_groups, is_test_results=True, is_real_time=False,
                                 do_print=False, save_folder=save_folder_name)

    elif res == 7: #For different trackers experiments
        sub_res = 8
        if sub_res == 0:
            folder_res = [
            ]
            save_folder_name = ""
        elif sub_res == 1:
            folder_res = [
                "output_final_DiffTrackers_test_res/HumanoidImitation_24-06-23-54-14_test_results/_05-07-22-38-00",
                "output_final_DiffTrackers_test_res/HumanoidImitation_24-06-23-54-14_test_results/_26-06-12-31-08",
                "output_final_DiffTrackers_test_res/HumanoidImitation_24-06-23-54-14_test_results/_26-06-12-49-38"

            ]
            save_folder_name = "output_final_DiffTrackers_test_res/HumanoidImitation_24-06-23-54-14_test_results"
        elif sub_res == 2:
            folder_res = [
                "output_final_DiffTrackers_test_res/HumanoidImitation_11-06-12-25-55_test_results/_07-07-14-50-08",
                "output_final_DiffTrackers_test_res/HumanoidImitation_11-06-12-25-55_test_results/_24-06-13-16-54",
                "output_final_DiffTrackers_test_res/HumanoidImitation_11-06-12-25-55_test_results/_25-06-01-49-32"

            ]
            save_folder_name = "output_final_DiffTrackers_test_res/HumanoidImitation_11-06-12-25-55_test_results"
        elif sub_res == 3:
            folder_res = [
                "output_final_DiffTrackers_test_res/HumanoidImitation_25-06-12-19-56_test_results/_06-07-10-02-07",
                "output_final_DiffTrackers_test_res/HumanoidImitation_25-06-12-19-56_test_results/_30-06-17-10-27",
                "output_final_DiffTrackers_test_res/HumanoidImitation_25-06-12-19-56_test_results/_30-06-17-28-52"
            ]
            save_folder_name = "output_final_DiffTrackers_test_res/HumanoidImitation_25-06-12-19-56_test_results"
        elif sub_res == 4:
            folder_res = [
                "output_final_DiffTrackers_test_res/HumanoidImitation_01-07-07-25-38_test_results/_05-07-23-13-09",
                "output_final_DiffTrackers_test_res/HumanoidImitation_01-07-07-25-38_test_results/_05-07-23-24-40",
                "output_final_DiffTrackers_test_res/HumanoidImitation_01-07-07-25-38_test_results/_06-07-00-51-34"
            ]
            save_folder_name = "output_final_DiffTrackers_test_res/HumanoidImitation_01-07-07-25-38_test_results"
        elif sub_res == 5:
            folder_res = [
                "output_final_DiffTrackers_test_res/HumanoidImitation_20-06-06-19-02_test_results/_07-07-15-25-11",
                "output_final_DiffTrackers_test_res/HumanoidImitation_20-06-06-19-02_test_results/_24-06-17-38-44",
                "output_final_DiffTrackers_test_res/HumanoidImitation_20-06-06-19-02_test_results/_25-06-17-13-43"

            ]
            save_folder_name = "output_final_DiffTrackers_test_res/HumanoidImitation_20-06-06-19-02_test_results"
        elif sub_res == 6:
            folder_res = [
                "output_final_DiffTrackers_test_res/HumanoidImitation_29-06-03-38-01_test_results/_06-07-21-42-43",
"output_final_DiffTrackers_test_res/HumanoidImitation_29-06-03-38-01_test_results/_06-07-21-54-56",
"output_final_DiffTrackers_test_res/HumanoidImitation_29-06-03-38-01_test_results/_06-07-23-28-51"
            ]
            save_folder_name = "output_final_DiffTrackers_test_res/HumanoidImitation_29-06-03-38-01_test_results"
        elif sub_res == 7:
            folder_res = [
                "output_final_DiffTrackers_test_res/HumanoidImitation_07-07-18-01-21_test_results/_10-07-23-45-29",
                "output_final_DiffTrackers_test_res/HumanoidImitation_07-07-18-01-21_test_results/_11-07-00-12-53",
                "output_final_DiffTrackers_test_res/HumanoidImitation_07-07-18-01-21_test_results/_11-07-03-33-08"
            ]
            save_folder_name = "output_final_DiffTrackers_test_res/HumanoidImitation_07-07-18-01-21_test_results"
        elif sub_res == 8:
            folder_res = [
                "output_final_DiffTrackers_test_res/HumanoidImitation_02-07-20-53-34_test_results/_11-07-04-26-31",
                "output_final_DiffTrackers_test_res/HumanoidImitation_02-07-20-53-34_test_results/_11-07-23-54-09",
                "output_final_DiffTrackers_test_res/HumanoidImitation_02-07-20-53-34_test_results/_12-07-00-20-32"

            ]
            save_folder_name = "output_final_DiffTrackers_test_res/HumanoidImitation_02-07-20-53-34_test_results"
        elif sub_res == 9:
            folder_res = [
                "output_final_DiffTrackers_test_res/HumanoidImitation_05-07-20-07-52_test_results/_11-07-05-21-17",
                "output_final_DiffTrackers_test_res/HumanoidImitation_05-07-20-07-52_test_results/_11-07-05-48-23",
                "output_final_DiffTrackers_test_res/HumanoidImitation_05-07-20-07-52_test_results/_11-07-09-14-50"
            ]
            save_folder_name = "output_final_DiffTrackers_test_res/HumanoidImitation_05-07-20-07-52_test_results"

        # grouping_keys_to_use = ['cfg_train', 'cfg_env_train', 'cfg_env_test', 'train_dataset', 'test_dataset',
        #                         'checkpoint']
        grouping_keys_to_use = ['cfg_env_train', 'cfg_train', 'test_dataset']
        val_to_name_dict = {
            'cfg_train': {
                "ase/data/cfg/train/rlg/common_ppo_humanoid_v3.yaml": "AltPar",
                "ase/data/cfg/train/rlg/common_ppo_humanoid.yaml": "QuestSimPar"
            },
            "cfg_env_train": {
                "ase/data/cfg/humanoid_imitation_vrh.yaml": "QuestSim(H+2C)",
                "ase/data/cfg/humanoid_imitation_vrhOne.yaml": "QuestSim(H)",
                "ase/data/cfg/humanoid_imitation_vrhm2Five.yaml": "QuestSim(H+2C+2F)",
                "ase/data/cfg/humanoid_imitation_vrh_pd_rewPenaltyAndReach.yaml": "Best(H+2C)",
                "ase/data/cfg/humanoid_imitation_vrhOne_pd_rewPenaltyAndReach.yaml": "Best(H)",
                "ase/data/cfg/humanoid_imitation_vrhm2Five_pd_rewPenaltyAndReach.yaml": "Best(H+2C+2F)",
                "ase/data/cfg/other_setup/humanoid_imitation_vrh_pd_rewPenaltyAndReach.yaml": "(1/72)Best(H+2C)",
                "ase/data/cfg/other_setup/humanoid_imitation_vrhOne_pd_rewPenaltyAndReach.yaml": "(1/72)Best(H)",
                "ase/data/cfg/other_setup/humanoid_imitation_vrhm2Five_pd_rewPenaltyAndReach.yaml": "(1/72)Best(H+2C+2F)"
            },
            'test_dataset': {
                "ase/data/motions/dataset_questsim_test.yaml": "DATASETTEST",
                "ase/data/motions/dataset_lafan_test.yaml": "LAFAN",
                "ase/data/motions/dataset_lafanlocomotion_test.yaml": "LAFANSIMILAR",

                "ase/data/motions/m2/dataset_questsim_test.yaml": "DATASETTEST",
                "ase/data/motions/m2/dataset_lafan_test.yaml": "LAFAN",
                "ase/data/motions/m2/dataset_lafanlocomotion_test.yaml": "LAFANSIMILAR"

            }
        }

        parameters, grouping_keys = extract_group_parameters(folder_res, is_test_results=True, is_real_time=False)
        for k in grouping_keys_to_use:
            assert k in grouping_keys
        folders_groups, groups_parameters = create_groups(parameters, folder_res, grouping_keys_to_use)
        names_groups = get_groups_names(groups_parameters, grouping_keys_to_use, val_to_name_dict)

        gr_res_list = [GroupResults(fols, is_test_results=True, is_real_time=False) for fols in
                       folders_groups]

        for gn in names_groups:
            print(gn)


        create_tables_for_groups(gr_res_list, groups_parameters, names_groups, is_test_results=True, is_real_time=False,
                                 do_print=False, save_folder=save_folder_name)






