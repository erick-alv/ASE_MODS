import os
from utils.motion_lib import MotionLib

def estimate_time(motion_file):
    _dof_body_ids = [1, 2, 4, 5, 8, 9, 12, 13, 14, 15, 16, 17]
    _dof_offsets = [0, 3, 6, 9, 10, 13, 14, 17, 18, 21, 24, 25, 28]

    ml = MotionLib(motion_file=motion_file,
              dof_body_ids=_dof_body_ids,
              dof_offsets=_dof_offsets,
              key_body_ids=[],
              device="cpu",verbose=False)
    return ml.get_total_length().item()

def get_all_motions_in_folder(path):
    m_files = os.listdir(path)
    m_files = [os.path.join(path, f) for f in m_files]
    return m_files


def main():
    motions_types = [
        #"balance",
        "conversation",
        # "dance",
        # "jump",
        # "locomotion",
        "conversation"#,
        # "balance",
        # "locomotion"
    ]
    paths_to_motions = [
        # "ase/data/motions/cmu_motions_retargeted/180/balance",
        "ase/data/motions/cmu_motions_retargeted/180/conversation",
        # "ase/data/motions/cmu_motions_retargeted/180/dance",
        # "ase/data/motions/cmu_motions_retargeted/180/jump",
        # "ase/data/motions/cmu_motions_retargeted/180/locomotion",
        "ase/data/motions/zeggs_motions_retargeted/180/conversation"#,
        # "ase/data/motions/bandai_namco_motions_retargeted/180/balance",
        # "ase/data/motions/bandai_namco_motions_retargeted/180/locomotion"
    ]
    for i, m_type in enumerate(motions_types):
        m_files = get_all_motions_in_folder(paths_to_motions[i])
        durations = [estimate_time(mf) for mf in m_files]
        duration_sum = sum(durations)
        print(f"The motion type {m_type} has a duration of {duration_sum} seconds")
    
if __name__ == "__main__":
    main()