import os
import json
import glob

from poselib.skeleton.skeleton3d import SkeletonTree, SkeletonState, SkeletonMotion
from poselib.visualization.common import plot_skeleton_state, plot_skeleton_motion_interactive
from scipy.spatial.transform import Rotation
import numpy as np
from poselib.core.rotation3d import *


# TODO the estimation of the tpose and rotation is not correct yet. But is enough for estimating the scale
def calc_adaptions_from_tpose(target_tpose, other_tpose, motion_type):
    target_ids = [target_tpose.skeleton_tree.index("head"), target_tpose.skeleton_tree.index("right_foot")]
    target_positions = [target_tpose.global_translation[i] for i in target_ids]
    target_height = target_positions[0][2] - target_positions[1][2]
    target_vecids = [target_tpose.skeleton_tree.index("pelvis"), target_tpose.skeleton_tree.index("head"),
                     target_tpose.skeleton_tree.index("left_hand"), target_tpose.skeleton_tree.index("right_hand"),
                     target_tpose.skeleton_tree.index("left_foot"), target_tpose.skeleton_tree.index("right_foot")]
    target_orvecs = [target_tpose.global_translation[i] - target_tpose.global_translation[target_vecids[0]] for i in
                     target_vecids[1:]]
    target_orvecs = np.array([x.numpy() for x in target_orvecs])

    if motion_type == "sfu":
        ids = [other_tpose.skeleton_tree.index("Head"), other_tpose.skeleton_tree.index("RightFoot")]
        positions = [other_tpose.global_translation[i] for i in ids]
        height = positions[0][1] - positions[1][1]
        scale = target_height / height

        vecids = [other_tpose.skeleton_tree.index("Hips"), other_tpose.skeleton_tree.index("Head"),
                  other_tpose.skeleton_tree.index("LeftHand"), other_tpose.skeleton_tree.index("RightHand"),
                  other_tpose.skeleton_tree.index("LeftFoot"), other_tpose.skeleton_tree.index("RightFoot")]
        orvecs = [other_tpose.global_translation[i] - other_tpose.global_translation[vecids[0]] for i in vecids[1:]]
        orvecs = np.array([x.numpy() for x in orvecs])
        rot, _ = Rotation.align_vectors(target_orvecs, orvecs)
    elif motion_type == "cmu":
        ids = [other_tpose.skeleton_tree.index("Head"), other_tpose.skeleton_tree.index("RightFoot")]
        positions = [other_tpose.global_translation[i] for i in ids]
        height = positions[0][2] - positions[1][2]
        scale = target_height / height

        vecids = [other_tpose.skeleton_tree.index("Hips"), other_tpose.skeleton_tree.index("Head"),
                  other_tpose.skeleton_tree.index("LeftHand"), other_tpose.skeleton_tree.index("RightHand"),
                  other_tpose.skeleton_tree.index("LeftFoot"), other_tpose.skeleton_tree.index("RightFoot")]
        orvecs = [other_tpose.global_translation[i] - other_tpose.global_translation[vecids[0]] for i in vecids[1:]]
        orvecs = np.array([x.numpy() for x in orvecs])

        #in this case we use just rotation around z axis as measure
        use_orvecs = orvecs.copy()
        use_orvecs[:, 2] = 0.
        use_target_orvecs = target_orvecs.copy()
        use_target_orvecs[:, 2] = 0.

        rot, _ = Rotation.align_vectors(use_target_orvecs, use_orvecs)

    elif motion_type == "lafan":
        ids = [other_tpose.skeleton_tree.index("Head"), other_tpose.skeleton_tree.index("RightFoot")]
        positions = [other_tpose.global_translation[i] for i in ids]
        height = positions[0][2] - positions[1][2]
        scale = target_height / height

        vecids = [other_tpose.skeleton_tree.index("Hips"), other_tpose.skeleton_tree.index("Head"),
                  other_tpose.skeleton_tree.index("LeftHand"), other_tpose.skeleton_tree.index("RightHand"),
                  other_tpose.skeleton_tree.index("LeftFoot"), other_tpose.skeleton_tree.index("RightFoot")]
        orvecs = [other_tpose.global_translation[i] - other_tpose.global_translation[vecids[0]] for i in vecids[1:]]
        orvecs = np.array([x.numpy() for x in orvecs])
        rot, _ = Rotation.align_vectors(target_orvecs, orvecs)
    else:
        raise Exception(f"the motion type {motion_type} does no exist")

    return scale, rot


def estimate_tpose_from_motion(motion, motion_type):
    if motion_type == "sfu":
        skeleton = motion.skeleton_tree
        zero_pose = SkeletonState.zero_pose(skeleton)

        # adjust pose into a T Pose
        local_rotation = zero_pose.local_rotation
        local_rotation[skeleton.index("LeftArm")] = quat_mul(
            quat_from_angle_axis(angle=torch.tensor([90.0]), axis=torch.tensor([0.0, 0.0, 1.0]), degree=True),
            local_rotation[skeleton.index("LeftArm")]
        )
        local_rotation[skeleton.index("RightArm")] = quat_mul(
            quat_from_angle_axis(angle=torch.tensor([-90.0]), axis=torch.tensor([0.0, 0.0, 1.0]), degree=True),
            local_rotation[skeleton.index("RightArm")]
        )
        return zero_pose
    elif motion_type == "cmu":
        skeleton = motion.skeleton_tree
        other_r = motion.local_rotation[0]
        original_t = motion.local_translation[0][0]

        other_t = torch.zeros(3, dtype=skeleton.local_translation.dtype)
        other_t[2] += original_t[2]

        other_pose = SkeletonState.from_rotation_and_root_translation(skeleton_tree=skeleton,
                                                                      r=other_r,
                                                                      t=other_t,
                                                                      is_local=True)
        return other_pose
    elif motion_type == "lafan":
        skeleton = motion.skeleton_tree
        other_r = motion.local_rotation[0]
        origin_glob_t = motion.global_translation[0][0]
        foot_id = skeleton.index("RightFoot")
        foot_glob_t = motion.global_translation[0][foot_id]
        diff = origin_glob_t - foot_glob_t
        #original_t = motion.local_translation[0][0]
        other_t = torch.zeros(3, dtype=skeleton.local_translation.dtype)
        #other_t[2] += original_t[2]
        other_t[2] += diff[2]
        other_pose = SkeletonState.from_rotation_and_root_translation(skeleton_tree=skeleton,
                                                                      r=other_r,
                                                                      t=other_t,
                                                                      is_local=True)
        return other_pose


        # adjust pose into a T Pose
        # local_rotation = zero_pose.local_rotation
        # local_rotation[skeleton.index("LeftArm")] = quat_mul(
        #     quat_from_angle_axis(angle=torch.tensor([90.0]), axis=torch.tensor([0.0, 0.0, 1.0]), degree=True),
        #     local_rotation[skeleton.index("LeftArm")]
        # )
        # return zero_pose
    else:
        raise Exception(f"the motion {motion_type} type does no exist")


def main():
    target_tpose = SkeletonState.from_file('data/amp_humanoid_vrh_tpose.npy')
    # motion_type = "sfu"
    # folder = "data/sfu_temp/"
    # sfu_tpose = SkeletonState.from_file('data/sfu_tpose.npy')
    # scale, rot = calc_adaptions_from_tpose(target_tpose, sfu_tpose, motion_type)
    # files_list = glob.glob(folder + "*.fbx")
    
    # motion_type = "cmu"
    # folder= "data/cmu_temp/"
    cmu_tpose = SkeletonState.from_file('data/cmu_tpose.npy')
    plot_skeleton_state(cmu_tpose)
    # scale, rot = calc_adaptions_from_tpose(target_tpose, cmu_tpose, motion_type)
    # files_list = glob.glob(folder + "*.fbx")

    motion_type = "lafan"
    files_list = ['../../../lafan1_fbx/aiming1_subject4.fbx']#, '../../../lafan1_fbx/ground1_subject1.fbx']

    print("******************************")

    save_tpose_file = True

    for fbx_file in files_list:

        print(fbx_file)
        # import fbx file - make sure to provide a valid joint name for root_joint
        motion = SkeletonMotion.from_fbx(
            fbx_file_path=fbx_file,
            root_joint="Hips",
            fps=60
        )

        other_tpose = estimate_tpose_from_motion(motion, motion_type)
        plot_skeleton_state(other_tpose)

        scale, rot = calc_adaptions_from_tpose(target_tpose, other_tpose, motion_type)
        print(f"scale: {scale}")
        print(f"rot: {rot.as_quat()}")

        if save_tpose_file:
            base = os.path.basename(fbx_file)[:-4]
            other_tpose.to_file(os.path.join("data", base + "_tpose.npy"))


        print("______________________________")


    print("Done")


if __name__ == "__main__":
    target_tpose = SkeletonState.from_file('data/amp_humanoid_vrh_tpose.npy')
    plot_skeleton_state(target_tpose)
    main()




