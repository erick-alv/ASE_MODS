import yaml
from utils.motion_lib import MotionLib
import argparse


motion_types = ["locomotion", "balance", "conversation", "dance", "jump", "other"]


def estimate_time(single_motion_file):
    if "retargeted_m2/" in single_motion_file:
        _dof_body_ids = [2, 3, 5, 6, 9, 10, 13, 14, 15, 17, 18, 19]
    else:
        _dof_body_ids = [1, 2, 4, 5, 8, 9, 12, 13, 14, 15, 16, 17]
    _dof_offsets = [0, 3, 6, 9, 10, 13, 14, 17, 18, 21, 24, 25, 28]

    # ml = MotionLib(motion_file=single_motion_file,
    #           dof_body_ids=_dof_body_ids,
    #           dof_offsets=_dof_offsets,
    #           key_body_ids=[],
    #           device="cpu", verbose=False)
    ml = MotionLib(motion_file=single_motion_file,
                   dof_body_ids=_dof_body_ids,
                   dof_offsets=_dof_offsets,
                   key_body_ids=[],
                   device="cuda:0", verbose=False)
    return ml.get_total_length().item()


def estimate_dataset_times(dataset_file, keep_running=False, do_all=False):
    if do_all:
        if "m2" in dataset_file:
            _dof_body_ids = [2, 3, 5, 6, 9, 10, 13, 14, 15, 17, 18, 19]
        else:
            _dof_body_ids = [1, 2, 4, 5, 8, 9, 12, 13, 14, 15, 16, 17]
        _dof_offsets = [0, 3, 6, 9, 10, 13, 14, 17, 18, 21, 24, 25, 28]
        ml = MotionLib(motion_file=dataset_file,
                       dof_body_ids=_dof_body_ids,
                       dof_offsets=_dof_offsets,
                       key_body_ids=[],
                       device="cuda:0")
    else:
        with open(dataset_file, 'r') as f:
            motions_dataset = yaml.load(f, Loader=yaml.SafeLoader)

        motions_types_list = {}
        for mt in motion_types:
            motions_types_list[mt] = []
        for element in motions_dataset["motions"]:
            for mt in motion_types:
                if "/"+mt+"/" in element["file"]:
                    motions_types_list[mt].append(element["file"])
            if "lafan" in element["file"]:#lafan does not classify motions
                motions_types_list["other"].append(element["file"])
        # for k,v in motions_types_list.items():
        #     print(f"motion type {k} has {len(v)}  files.")

        motion_time_total = 0.0
        all_motion_libs = []
        for k, v in motions_types_list.items():
            if len(v) > 0:
                durations = [estimate_time(mf) for mf in v]
                duration_sum = sum(durations)
            else:
                duration_sum = 0.0
            motion_time_total += duration_sum
            print(f"The motion type {k} has a duration of {duration_sum} seconds")
        print(f"In total the dataset has a duration of {motion_time_total} seconds")
    if keep_running:
        while True:
            print("running")
            if do_all:
                print(ml)





if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_file', type=str,
                        help='path to the dataset file',
                        required=True)
    parser.add_argument('--keep_running', action='store_true', help='keeps running in order to keep the files in memory')
    args = parser.parse_args()
    estimate_dataset_times(args.dataset_file, args.keep_running, do_all=True)

