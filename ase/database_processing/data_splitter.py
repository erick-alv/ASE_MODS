import os
import numpy as np
import yaml
import re
from utils.common_constants import HEIGHT_FOLDER_PATTERN


def separate_folders(path, split_fn, include_motion_types=[]):
    heights_folders = [f for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))]
    for folder in heights_folders:
        assert re.compile(HEIGHT_FOLDER_PATTERN).match(folder), "the folder's name does not have format for height"

    motion_type_folders = [f for f in os.listdir(os.path.join(path, heights_folders[0])) if os.path.isdir(os.path.join(path, heights_folders[0], f))]
    if len(motion_type_folders) > 0:
        train_files = []
        test_files = []
        for motion_type_folder in motion_type_folders:
            if len(include_motion_types) == 0 or motion_type_folder in include_motion_types:
                current_motion_files = get_motion_files(os.path.join(path, heights_folders[0], motion_type_folder))
                current_motion_files = [os.path.join(motion_type_folder, f) for f in current_motion_files]
                current_train_files, current_test_files = split_fn(current_motion_files)
                train_files.extend(current_train_files)
                test_files.extend(current_test_files)

    else:
        motion_files = get_motion_files(os.path.join(path, heights_folders[0]))
        train_files, test_files = split_fn(motion_files)
    return heights_folders, train_files, test_files


def get_motion_files(path):
    #return glob.glob(path + "/*.npy")
    return [f for f in os.listdir(path) if f.endswith(".npy")]


def create_dataset_yaml(motion_files_dict, filename):

    file_basename = os.path.basename(filename)
    file_dir_heights = filename + "_hs" + os.sep
    if not os.path.exists(file_dir_heights):
        os.makedirs(file_dir_heights, exist_ok=True)


    heights_yaml_files = []
    for key, motion_files in motion_files_dict.items():
        weight = 1.0 / len(motion_files)
        dict_file = {'motions': []}
        dict_file['wd_path'] = True
        for m in motion_files:
            dict_file['motions'].append({"file": m, "weight": weight})

        current_filename = os.path.join(file_dir_heights,  file_basename + "_" + key + ".yaml")

        with open(current_filename, 'w') as file:
            yaml.dump(dict_file, file)

        heights_yaml_files.append({"file": current_filename, "key": int(key)})

    all_heights_filename = filename + "_all_heights.yaml"
    with open(all_heights_filename, 'w') as mf:
        yaml.dump(heights_yaml_files, mf)


def create_splits_files(motion_files_folders, split_prc, path_to_data, dataset_file_name, include_motion_types=[]):
    assert len(split_prc) == 2
    assert split_prc[0] + split_prc[1] == 1
    all_train_files = {}
    all_test_files = {}
    ref_heights_folders = None
    np.random.seed(0)

    def separate_in_splits(motion_files):
        num_files = len(motion_files)
        if num_files > 0:
            np.random.shuffle(motion_files)
            tr, te = np.split(motion_files, [
                int(num_files * split_prc[0])
            ])
            return tr, te

    for motion_folder in motion_files_folders:

        heights_folders, train_files, test_files = separate_folders(
            os.path.join(path_to_data, motion_folder), separate_in_splits, include_motion_types=include_motion_types)
        if ref_heights_folders is None:
            ref_heights_folders = heights_folders
            for height_f in heights_folders:
                all_train_files[height_f] = []
                all_test_files[height_f] = []
        else:
            for height_f in heights_folders:
                assert height_f in ref_heights_folders
            for height_f in ref_heights_folders:
                assert height_f in heights_folders

        for height_f in heights_folders:
            all_train_files[height_f].extend(
                [os.path.join(path_to_data, motion_folder, height_f, f) for f in train_files]
            )
            all_test_files[height_f].extend(
                [os.path.join(path_to_data, motion_folder, height_f, f) for f in test_files]
            )


    # print("train")
    # print(all_train_files)
    # print("test")
    # print(all_test_files)
    if len(all_train_files) > 0:
        create_dataset_yaml(all_train_files, os.path.join(path_to_data, dataset_file_name+"_train"))
    if len(all_test_files) > 0:
        create_dataset_yaml(all_test_files, os.path.join(path_to_data, dataset_file_name + "_test"))



def main():
    path_to_data = "ase/data/motions/"

    database_num = 11

    training_motions_folders = ['cmu_motions_retargeted', 'zeggs_motions_retargeted', 'bandai_namco_motions_retargeted']
    training_split_prc = [0.9, 0.1]


    if database_num == 0:
        motion_files_folders = ['cmu_temp_retargeted']
        dataset_file_name = "dataset_temp"
        create_splits_files(motion_files_folders, [0.7, 0.3], path_to_data=path_to_data,
                            dataset_file_name=dataset_file_name)

    elif database_num == 1:
        #Database 1; as in QUESTSIM --> ignore dance and jump
        dataset_file_name = "dataset_questsim"
        include = ["locomotion", "balance", "conversation"]
        create_splits_files(training_motions_folders, training_split_prc, path_to_data=path_to_data,
                            dataset_file_name=dataset_file_name, include_motion_types=include)

    elif database_num == 2:
        # Database 2 all
        dataset_file_name = "dataset_all"
        include = ["locomotion", "balance", "conversation", "jump", "dance"]
        create_splits_files(training_motions_folders, training_split_prc, path_to_data=path_to_data,
                            dataset_file_name=dataset_file_name, include_motion_types=include)

    elif database_num == 3:
        # Database 3; just extra dance --> ignore jump
        dataset_file_name = "dataset_plusdance"
        include = ["locomotion", "balance", "conversation", "dance"]
        create_splits_files(training_motions_folders, training_split_prc, path_to_data=path_to_data,
                            dataset_file_name=dataset_file_name, include_motion_types=include)
    elif database_num == 4:
        # Database 4; just extra jump --> ignore dance
        dataset_file_name = "dataset_plusjump"
        include = ["locomotion", "balance", "conversation", "jump"]
        create_splits_files(training_motions_folders, training_split_prc, path_to_data=path_to_data,
                            dataset_file_name=dataset_file_name, include_motion_types=include)
    elif database_num == 5:
        #Database 5 Test ; just uses lafan
        motion_files_folders = ['lafan_motions_retargeted']
        dataset_file_name = "dataset_lafan"
        #here we need all
        create_splits_files(motion_files_folders, [0, 1], path_to_data=path_to_data,
                            dataset_file_name=dataset_file_name)
    elif database_num == 6:
        #just locomotion
        dataset_file_name = "dataset_locomotion"
        include = ["locomotion"]
        create_splits_files(training_motions_folders, training_split_prc, path_to_data=path_to_data,
                            dataset_file_name=dataset_file_name, include_motion_types=include)

    elif database_num == 7:
        # just balance
        dataset_file_name = "dataset_balance"
        include = ["balance"]
        create_splits_files(training_motions_folders, training_split_prc, path_to_data=path_to_data,
                            dataset_file_name=dataset_file_name, include_motion_types=include)

    elif database_num == 8:
        #just conversation
        dataset_file_name = "dataset_conversation"
        include = ["conversation"]
        create_splits_files(training_motions_folders, training_split_prc, path_to_data=path_to_data,
                            dataset_file_name=dataset_file_name, include_motion_types=include)
    elif database_num == 9:
        #just jump
        dataset_file_name = "dataset_jump"
        include = ["jump"]
        create_splits_files(training_motions_folders, training_split_prc, path_to_data=path_to_data,
                            dataset_file_name=dataset_file_name, include_motion_types=include)
    elif database_num == 10:
        #just dance
        dataset_file_name = "dataset_dance"
        include = ["dance"]
        create_splits_files(training_motions_folders, training_split_prc, path_to_data=path_to_data,
                            dataset_file_name=dataset_file_name, include_motion_types=include)


    elif database_num == 11:
        motion_files_folders = ['bandai_namco_motions_retargeted']
        dataset_file_name = "bn"
        #not using include will automatically includes all
        create_splits_files(motion_files_folders, [0.9, 0.1], path_to_data=path_to_data,
                            dataset_file_name=dataset_file_name)




if __name__=="__main__":
    main()
