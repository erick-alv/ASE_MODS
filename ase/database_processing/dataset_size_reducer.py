import yaml
import os
import random
import argparse

motion_classes = [
    "locomotion", "balance", "conversation", "jump", "dance"
]

def read_and_classify(dataset_file):
    with open(os.path.join(os.getcwd(), dataset_file), 'r') as f:
        motions_dict_list = yaml.load(f, Loader=yaml.SafeLoader)
    classified_dict = {

    }
    for mc in motion_classes:
        classified_dict[mc] = []
    for entry in motions_dict_list["motions"]:
        for mc in motion_classes:
            if "/"+mc+"/" in entry["file"]:
                classified_dict[mc].append(entry)
    return motions_dict_list, classified_dict


def reduce_number(approx_desired_size, classified_dict, keep_type_list=[]):
    to_keep = {}
    to_keep_size = 0
    can_eliminate = {}
    can_eliminate_size = 0
    for mc, fl in classified_dict.items():
        if mc in keep_type_list:
            to_keep[mc] = fl
            to_keep_size += len(fl)
        else:
            can_eliminate[mc] = fl
            can_eliminate_size += len(fl)
    n_del = can_eliminate_size + to_keep_size - approx_desired_size
    n_size = 0
    assert n_del <= can_eliminate_size
    for mc, fl in can_eliminate.items():
        c_del = len(fl) / can_eliminate_size * n_del
        c_del = int(c_del)
        for _ in range(c_del):
            fl.pop(random.randrange(len(fl)))
        can_eliminate[mc] = fl
        n_size += len(fl)
    to_keep.update(can_eliminate)
    return to_keep, n_size+to_keep_size


def create_file(original_filename, original_motions_dict_list, new_size, reduced_classified):
    all_motions_list = []
    for mc, fl in reduced_classified.items():
        all_motions_list.extend(fl)

    f_dict = {
        "motions": all_motions_list,
        "wd_path": original_motions_dict_list["wd_path"]
    }
    n_name = original_filename[:-5] + "_" + str(new_size) +".yaml"

    with open(n_name, 'w') as file:
        yaml.dump(f_dict, file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_file', type=str,
                        help='path to the dataset file ',
                        required=True)
    parser.add_argument('--desired_size',
                        type=int,
                        help='The desired number of motion in file',
                        required=True)
    parser.add_argument('--keep_class_list', nargs='+', default=[],
                        help='the motion class type that we do not want to eliminate',
                        required=False)
    args = parser.parse_args()
    dataset_file = args.dataset_file
    motions_dict_list, classified_dict = read_and_classify(dataset_file)
    reduced_classified, new_size = reduce_number(args.desired_size, classified_dict, keep_type_list=args.keep_class_list)
    create_file(dataset_file, motions_dict_list, new_size, reduced_classified)