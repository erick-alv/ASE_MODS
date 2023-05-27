import os
import numpy as np
import yaml
import re
from utils.common_constants import HEIGHT_FOLDER_PATTERN


def separate_folders(path, split_fn, include_motion_types=[]):
    motion_type_folders = [f for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))]
    if len(motion_type_folders) > 0:
        train_files = []
        test_files = []
        for motion_type_folder in motion_type_folders:
            if len(include_motion_types) == 0 or motion_type_folder in include_motion_types:
                current_motion_files = get_motion_files(os.path.join(path, motion_type_folder))
                current_motion_files = [os.path.join(motion_type_folder, f) for f in current_motion_files]
                current_train_files, current_test_files = split_fn(current_motion_files)
                train_files.extend(current_train_files)
                test_files.extend(current_test_files)

    else:
        motion_files = get_motion_files(path)
        train_files, test_files = split_fn(motion_files)
    return train_files, test_files


def get_motion_files(path):
    return [f for f in os.listdir(path) if f.endswith(".npy")]


def create_dataset_yaml(motion_files, filename, mt_sample_factor_dict):

    def get_motion_type(filepath):
        path_parts = filepath.split(os.sep)
        if len(path_parts) < 2:
            return None
        else:
            return path_parts[-2]

    weight = 1.0 / len(motion_files)
    dict_file = {'motions': []}
    dict_file['wd_path'] = True
    for m in motion_files:
        motion_type = get_motion_type(m)
        if motion_type is None or motion_type not in mt_sample_factor_dict.keys():
            extra_sample_factor = 1
        else:
            extra_sample_factor = mt_sample_factor_dict[motion_type]
        final_weight = extra_sample_factor * weight
        dict_file['motions'].append({"file": m, "weight": final_weight})

    current_filename = filename + ".yaml"

    with open(current_filename, 'w') as file:
        yaml.dump(dict_file, file)




def create_splits_files(motion_files_folders, split_prc, path_to_data, dataset_file_name, include_motion_types=[],
                        mt_sample_factor_dict={}):
    assert len(split_prc) == 2
    assert split_prc[0] + split_prc[1] == 1
    all_train_files = []
    all_test_files = []
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
        path_to_folder = os.path.join(path_to_data, motion_folder)
        # heights_folders, train_files, test_files = separate_folders(path_to_folder, separate_in_splits,
        #                                                             include_motion_types=include_motion_types)
        train_files, test_files = separate_folders(path_to_folder, separate_in_splits,
                                                   include_motion_types=include_motion_types)
        all_train_files.extend(
            [os.path.join(path_to_data, motion_folder, f) for f in train_files]
        )
        all_test_files.extend(
            [os.path.join(path_to_data, motion_folder, f) for f in test_files]
        )


    # print("train")
    # print(all_train_files)
    # print("test")
    # print(all_test_files)
    if len(all_train_files) > 0:
        create_dataset_yaml(all_train_files, os.path.join(path_to_data, dataset_file_name+"_train"),
                            mt_sample_factor_dict)
    if len(all_test_files) > 0:
        create_dataset_yaml(all_test_files, os.path.join(path_to_data, dataset_file_name + "_test"),
                            mt_sample_factor_dict)


def main():
    path_to_data = "ase/data/motions/"

    database_num = 11

    training_motions_folders = ['cmu_motions_retargeted/180',
                                'zeggs_motions_retargeted/180',
                                'bandai_namco_motions_retargeted/180']
    training_split_prc = [0.9, 0.1]
    #we use this in an attempt to sample more of the motions from conversations
    # because we have fewer files than the others (but each of conversation has a longer duration)
    # training_sample_factor_dict = {
    #     "locomotion":1,
    #     "balance":1,
    #     "conversation":1.5,
    #     "jump":1,
    #     "dance":1
    # }
    #also considering that we have fewer locomotion movements than in questsim
    #Considering that we have few from jump and dance
    training_sample_factor_dict = {
        "locomotion": 2,
        "balance": 1,
        "conversation": 22,
        "jump": 11,
        "dance": 20
    }


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
                            dataset_file_name=dataset_file_name, include_motion_types=include,
                            mt_sample_factor_dict=training_sample_factor_dict)

    elif database_num == 2:
        # Database 2 all
        dataset_file_name = "dataset_all"
        include = ["locomotion", "balance", "conversation", "jump", "dance"]
        create_splits_files(training_motions_folders, training_split_prc, path_to_data=path_to_data,
                            dataset_file_name=dataset_file_name, include_motion_types=include,
                            mt_sample_factor_dict=training_sample_factor_dict)

    elif database_num == 3:
        # Database 3; just extra dance --> ignore jump
        dataset_file_name = "dataset_plusdance"
        include = ["locomotion", "balance", "conversation", "dance"]
        create_splits_files(training_motions_folders, training_split_prc, path_to_data=path_to_data,
                            dataset_file_name=dataset_file_name, include_motion_types=include,
                            mt_sample_factor_dict=training_sample_factor_dict)
    elif database_num == 4:
        # Database 4; just extra jump --> ignore dance
        dataset_file_name = "dataset_plusjump"
        include = ["locomotion", "balance", "conversation", "jump"]
        create_splits_files(training_motions_folders, training_split_prc, path_to_data=path_to_data,
                            dataset_file_name=dataset_file_name, include_motion_types=include,
                            mt_sample_factor_dict=training_sample_factor_dict)
    elif database_num == 5:
        #Database 5 Test ; just uses lafan
        motion_files_folders = ['lafan_motions_retargeted/180']
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
        #just dance
        dataset_file_name = "dataset_loc_bal"
        include = ["locomotion", "balance"]
        create_splits_files(training_motions_folders, training_split_prc, path_to_data=path_to_data,
                            dataset_file_name=dataset_file_name, include_motion_types=include,
                            mt_sample_factor_dict=training_sample_factor_dict)





if __name__=="__main__":
    main()
