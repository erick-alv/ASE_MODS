import os
import numpy as np
import yaml


def get_sub_folders(path):
    return [f for f in os.listdir(path) if os.path.isdir(os.path.join(path,f))]


def get_motion_files(path):
    #return glob.glob(path + "/*.npy")
    return [f for f in os.listdir(path) if f.endswith(".npy")]

def create_dataset_yaml(motion_file_list, filename):
    weight = 1.0 / len(motion_file_list)
    dict_file = {'motions': []}
    for m in motion_file_list:
        dict_file['motions'].append({"file": m, "weight": weight})

    with open(filename, 'w') as file:
        documents = yaml.dump(dict_file, file)


def create_splits_files(data_folders, split_prc, common_path, dataset_file_name):
    assert len(split_prc) == 3
    assert split_prc[0] + split_prc[1] + split_prc[2] == 1
    train_files = []
    validation_files = []
    test_files = []

    np.random.seed(0)

    def separate_in_splits(motion_files, parent_folder):
        num_files = len(motion_files)
        if num_files > 0:
            np.random.shuffle(motion_files)
            tr, val, te = np.split(motion_files, [
                int(num_files*split_prc[0]),
                int(num_files*(split_prc[0]+split_prc[1]))
            ])
            tr = [os.path.join(parent_folder, f) for f in tr]
            val = [os.path.join(parent_folder, f) for f in val]
            te = [os.path.join(parent_folder, f) for f in te]
            train_files.extend(tr)
            validation_files.extend(val)
            test_files.extend(te)

    for data_folder in data_folders:
        subfolfders = get_sub_folders(data_folder)
        if len(subfolfders) > 0:
            for subfolfder in subfolfders:
                print(subfolfder)
                motion_files = get_motion_files(os.path.join(data_folder, subfolfder))
                separate_in_splits(motion_files, parent_folder=os.path.join(data_folder[len(common_path):], subfolfder))

        else:
            motion_files = get_motion_files(data_folder)
            separate_in_splits(motion_files, parent_folder=data_folder[len(common_path):])

    # print("train")
    # print(train_files)
    # print("val")
    # print(validation_files)
    # print("test")
    # print(test_files)
    if len(train_files) > 0:
        create_dataset_yaml(train_files, os.path.join(common_path, dataset_file_name+"_train.yaml"))
    if len(validation_files) > 0:
        create_dataset_yaml(validation_files, os.path.join(common_path, dataset_file_name + "_val.yaml"))
    if len(test_files) > 0:
        create_dataset_yaml(test_files, os.path.join(common_path, dataset_file_name + "_test.yaml"))



def main():
    data_folders = ['data/motions/cmu_motions_retargeted/', 'data/motions/sfu_temp_retargeted/', 'data/motions/zeggs_temp_retargeted/']
    dataset_file_name = "dataset_imit"
    common_path = "data/motions/"
    create_splits_files(data_folders, [0.8, 0.1, 0.1], common_path=common_path, dataset_file_name=dataset_file_name)



if __name__=="__main__":
    main()
