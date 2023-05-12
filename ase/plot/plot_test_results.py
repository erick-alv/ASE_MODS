from data_load import extract_group_parameters, create_groups, GroupResults
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

configs_to_names = {
    "{'track': ['headset', 'right_controller', 'left_controller'], 'algo': 'common', 'train_dataset': 'ase/data/motions/dataset_cmu_07_01_all_heights.yaml', 'real_time': False, 'test_dataset': ''}": "A",
    "{'track': ['headset', 'right_controller', 'left_controller'], 'algo': 'common', 'train_dataset': 'ase/data/motions/dataset_cmu_07_01_all_heights_mock2.yaml', 'real_time': False, 'test_dataset': ''}":"B"
}


colors = ["orange", "blue", "red", "green", "black", "purple", "gold", "cyan", "lime", "brown"]


def smooth(df: pd.DataFrame, smooth_factor):
    #we use 1-smooth factor to have similar to tensorboard whn usinf smooth
    return df.ewm(alpha=(1-smooth_factor), adjust=True).mean()


def plot_groups_val(groups_res_list, groups_attrs, val_name, smooth_factor=None,
                    filename=None, do_plot=True,
                    title=None, xlabel=None, ylabel=None, legend_outside=False, plot_std=True):
    def plot_single(df_val, df_std, number_i):
        if smooth_factor is not None:
            df_val = smooth(df_val, smooth_factor)
            df_std = smooth(df_std, smooth_factor)

        x_axis_vals_array = df_val.index.values
        vals_array = df_val.iloc[:, 0].values
        std_array = df_std.iloc[:, 0].values

        attr_str = str(groups_attrs[number_i])
        if attr_str in configs_to_names.keys():
            label_name = configs_to_names[attr_str]
        else:
            print(attr_str)
            label_name = attr_str

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


def create_plots_for_groups(groups_res_list, groups_attrs, save_folder=None, is_test_results=False, is_real_time=False,
                            do_plot=True, extra_plot_title_name=None):
    if is_test_results:
        if is_real_time:
            val_names = ["real_time-ar_aj_pos_error_track", "real_time-ar_aj_rot_error_track",
                         'real_time-ar_pos_error_track/headset',  'real_time-ar_rot_error_track/headset']
            smooth_factor_list = [0.1, 0.1, 0.1, 0.1]
            to_title = {
                "real_time-ar_aj_pos_error_track": "MHPE",
                "real_time-ar_aj_rot_error_track": "MHRE",
                'real_time-ar_pos_error_track/headset': "MHPE (headset)",
                'real_time-ar_rot_error_track/headset': "MHRE (headset)"
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
            # todo rewards of motions
            # todo create a set of motions for those that we want to plot the rewards
            #  (this will depend on test dataset)
            val_names = ['07_01_cmu_amp.npy-ar_reward']
            smooth_factor_list = [0.2]
            to_title = {
                '07_01_cmu_amp.npy-ar_reward': 'Reward 07_01_cmu_amp',
            }

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

        plot_groups_val(groups_res_list, groups_attrs, val_name=val_name, smooth_factor=smooth_factor,
                        title=title, xlabel=xlabel, ylabel=ylabel,
                        legend_outside=False, plot_std=True, do_plot=do_plot,
                        filename=filename)

def create_tables_for_groups(groups_res_list, groups_attrs, save_folder=None, is_test_results=False,
                             is_real_time=False, do_print=False):

    def fill_table_dict(table_dict):
        for i, gr_res in enumerate(groups_res_list):
            for key in table_dict.keys():
                if key == "method":
                    table_dict[key].append(str(groups_attrs[i]))
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
        with open(filename, "w") as f:
            f.write(df.to_latex(float_format="{:.2f}".format))

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
            'real_time-ar_as_aj_pos_error_track': 'MHPE [cm]',
            'real_time-ar_as_aj_rot_error_track': 'MHRE [deg]',
            'real_time-ar_as_pos_error_track/headset': 'MHPE [cm] (headset)',
            'real_time-ar_as_pos_error_track/right_controller': 'MHPE [cm] (right controller)',
            'real_time-ar_as_pos_error_track/left_controller': 'MHPE [cm] (left controller)',
            'real_time-ar_as_rot_error_track/headset': 'MHRE [deg] (headset)',
            'real_time-ar_as_rot_error_track/right_controller': 'MHRE [deg] (right controller)',
            'real_time-ar_as_rot_error_track/left_controller': 'MHRE [deg] (left controller)'
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
            'ar_am_as_aj_pos_error': [],
            'ar_am_as_aj_pos_error_track': [],
            'ar_am_as_aj_rot_error': [],
            'ar_am_as_aj_rot_error_track': [],
            'ar_am_as_sip': []

        }
        rewards_table_dict = {
            "method": [],
            'ar_am_reward': [],
            'ar_am_weighted_reward': []
        }
        sim_name_to_table_name = {
            'ar_am_as_aj_jitter': 'Jitter [km/s³]',
            'ar_am_as_aj_pos_error': 'MPJPE [cm]',
            'ar_am_as_aj_pos_error_track': 'MHPE [cm]',
            'ar_am_as_aj_rot_error': 'MPJRE [deg]',
            'ar_am_as_aj_rot_error_track': 'MHRE [deg]',
            'ar_am_as_sip': 'SIP [deg]',
            'ar_am_reward': 'reward',
            'ar_am_weighted_reward': 'weighted reward'
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

if __name__ == "__main__":

    #to save just specify in which folder; then function decides which name to give

    res = 2
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

