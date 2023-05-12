import argparse
import os
import json
import numpy as np

from poselib.core.rotation3d import *
from poselib.skeleton.skeleton3d import SkeletonTree, SkeletonState, SkeletonMotion
from poselib.visualization.common import plot_skeleton_state, plot_skeleton_motion_interactive
from fbx_tpose import calc_adaptions_from_tpose, estimate_tpose_from_motion
import glob
import re

"""
This scripts shows how to retarget a motion clip from the source skeleton to a target skeleton.
Data required for retargeting are stored in a retarget config dictionary as a json file. This file contains:
  - source_motion: a SkeletonMotion npy format representation of a motion sequence. The motion clip should use the same skeleton as the source T-Pose skeleton.
  - target_motion_path: path to save the retargeted motion to
  - source_tpose: a SkeletonState npy format representation of the source skeleton in it's T-Pose state
  - target_tpose: a SkeletonState npy format representation of the target skeleton in it's T-Pose state (pose should match source T-Pose)
  - joint_mapping: mapping of joint names from source to target
  - rotation: root rotation offset from source to target skeleton (for transforming across different orientation axes), represented as a quaternion in XYZW order.
  - scale: scale offset from source to target skeleton
"""

def project_joints(motion):
    right_upper_arm_id = motion.skeleton_tree._node_indices["right_upper_arm"]
    right_lower_arm_id = motion.skeleton_tree._node_indices["right_lower_arm"]
    right_hand_id = motion.skeleton_tree._node_indices["right_hand"]
    left_upper_arm_id = motion.skeleton_tree._node_indices["left_upper_arm"]
    left_lower_arm_id = motion.skeleton_tree._node_indices["left_lower_arm"]
    left_hand_id = motion.skeleton_tree._node_indices["left_hand"]
    
    right_thigh_id = motion.skeleton_tree._node_indices["right_thigh"]
    right_shin_id = motion.skeleton_tree._node_indices["right_shin"]
    right_foot_id = motion.skeleton_tree._node_indices["right_foot"]
    left_thigh_id = motion.skeleton_tree._node_indices["left_thigh"]
    left_shin_id = motion.skeleton_tree._node_indices["left_shin"]
    left_foot_id = motion.skeleton_tree._node_indices["left_foot"]
    
    device = motion.global_translation.device

    # right arm
    right_upper_arm_pos = motion.global_translation[..., right_upper_arm_id, :]
    right_lower_arm_pos = motion.global_translation[..., right_lower_arm_id, :]
    right_hand_pos = motion.global_translation[..., right_hand_id, :]
    right_shoulder_rot = motion.local_rotation[..., right_upper_arm_id, :]
    right_elbow_rot = motion.local_rotation[..., right_lower_arm_id, :]
    
    right_arm_delta0 = right_upper_arm_pos - right_lower_arm_pos
    right_arm_delta1 = right_hand_pos - right_lower_arm_pos
    right_arm_delta0 = right_arm_delta0 / torch.norm(right_arm_delta0, dim=-1, keepdim=True)
    right_arm_delta1 = right_arm_delta1 / torch.norm(right_arm_delta1, dim=-1, keepdim=True)
    right_elbow_dot = torch.sum(-right_arm_delta0 * right_arm_delta1, dim=-1)
    right_elbow_dot = torch.clamp(right_elbow_dot, -1.0, 1.0)
    right_elbow_theta = torch.acos(right_elbow_dot)
    right_elbow_q = quat_from_angle_axis(-torch.abs(right_elbow_theta), torch.tensor(np.array([[0.0, 1.0, 0.0]]), 
                                            device=device, dtype=torch.float32))
    
    right_elbow_local_dir = motion.skeleton_tree.local_translation[right_hand_id]
    right_elbow_local_dir = right_elbow_local_dir / torch.norm(right_elbow_local_dir)
    right_elbow_local_dir_tile = torch.tile(right_elbow_local_dir.unsqueeze(0), [right_elbow_rot.shape[0], 1])
    right_elbow_local_dir0 = quat_rotate(right_elbow_rot, right_elbow_local_dir_tile)
    right_elbow_local_dir1 = quat_rotate(right_elbow_q, right_elbow_local_dir_tile)
    right_arm_dot = torch.sum(right_elbow_local_dir0 * right_elbow_local_dir1, dim=-1)
    right_arm_dot = torch.clamp(right_arm_dot, -1.0, 1.0)
    right_arm_theta = torch.acos(right_arm_dot)
    right_arm_theta = torch.where(right_elbow_local_dir0[..., 1] <= 0, right_arm_theta, -right_arm_theta)
    right_arm_q = quat_from_angle_axis(right_arm_theta, right_elbow_local_dir.unsqueeze(0))
    right_shoulder_rot = quat_mul(right_shoulder_rot, right_arm_q)
    
    # left arm
    left_upper_arm_pos = motion.global_translation[..., left_upper_arm_id, :]
    left_lower_arm_pos = motion.global_translation[..., left_lower_arm_id, :]
    left_hand_pos = motion.global_translation[..., left_hand_id, :]
    left_shoulder_rot = motion.local_rotation[..., left_upper_arm_id, :]
    left_elbow_rot = motion.local_rotation[..., left_lower_arm_id, :]
    
    left_arm_delta0 = left_upper_arm_pos - left_lower_arm_pos
    left_arm_delta1 = left_hand_pos - left_lower_arm_pos
    left_arm_delta0 = left_arm_delta0 / torch.norm(left_arm_delta0, dim=-1, keepdim=True)
    left_arm_delta1 = left_arm_delta1 / torch.norm(left_arm_delta1, dim=-1, keepdim=True)
    left_elbow_dot = torch.sum(-left_arm_delta0 * left_arm_delta1, dim=-1)
    left_elbow_dot = torch.clamp(left_elbow_dot, -1.0, 1.0)
    left_elbow_theta = torch.acos(left_elbow_dot)
    left_elbow_q = quat_from_angle_axis(-torch.abs(left_elbow_theta), torch.tensor(np.array([[0.0, 1.0, 0.0]]), 
                                        device=device, dtype=torch.float32))

    left_elbow_local_dir = motion.skeleton_tree.local_translation[left_hand_id]
    left_elbow_local_dir = left_elbow_local_dir / torch.norm(left_elbow_local_dir)
    left_elbow_local_dir_tile = torch.tile(left_elbow_local_dir.unsqueeze(0), [left_elbow_rot.shape[0], 1])
    left_elbow_local_dir0 = quat_rotate(left_elbow_rot, left_elbow_local_dir_tile)
    left_elbow_local_dir1 = quat_rotate(left_elbow_q, left_elbow_local_dir_tile)
    left_arm_dot = torch.sum(left_elbow_local_dir0 * left_elbow_local_dir1, dim=-1)
    left_arm_dot = torch.clamp(left_arm_dot, -1.0, 1.0)
    left_arm_theta = torch.acos(left_arm_dot)
    left_arm_theta = torch.where(left_elbow_local_dir0[..., 1] <= 0, left_arm_theta, -left_arm_theta)
    left_arm_q = quat_from_angle_axis(left_arm_theta, left_elbow_local_dir.unsqueeze(0))
    left_shoulder_rot = quat_mul(left_shoulder_rot, left_arm_q)
    
    # right leg
    right_thigh_pos = motion.global_translation[..., right_thigh_id, :]
    right_shin_pos = motion.global_translation[..., right_shin_id, :]
    right_foot_pos = motion.global_translation[..., right_foot_id, :]
    right_hip_rot = motion.local_rotation[..., right_thigh_id, :]
    right_knee_rot = motion.local_rotation[..., right_shin_id, :]
    
    right_leg_delta0 = right_thigh_pos - right_shin_pos
    right_leg_delta1 = right_foot_pos - right_shin_pos
    right_leg_delta0 = right_leg_delta0 / torch.norm(right_leg_delta0, dim=-1, keepdim=True)
    right_leg_delta1 = right_leg_delta1 / torch.norm(right_leg_delta1, dim=-1, keepdim=True)
    right_knee_dot = torch.sum(-right_leg_delta0 * right_leg_delta1, dim=-1)
    right_knee_dot = torch.clamp(right_knee_dot, -1.0, 1.0)
    right_knee_theta = torch.acos(right_knee_dot)
    right_knee_q = quat_from_angle_axis(torch.abs(right_knee_theta), torch.tensor(np.array([[0.0, 1.0, 0.0]]), 
                                        device=device, dtype=torch.float32))
    
    right_knee_local_dir = motion.skeleton_tree.local_translation[right_foot_id]
    right_knee_local_dir = right_knee_local_dir / torch.norm(right_knee_local_dir)
    right_knee_local_dir_tile = torch.tile(right_knee_local_dir.unsqueeze(0), [right_knee_rot.shape[0], 1])
    right_knee_local_dir0 = quat_rotate(right_knee_rot, right_knee_local_dir_tile)
    right_knee_local_dir1 = quat_rotate(right_knee_q, right_knee_local_dir_tile)
    right_leg_dot = torch.sum(right_knee_local_dir0 * right_knee_local_dir1, dim=-1)
    right_leg_dot = torch.clamp(right_leg_dot, -1.0, 1.0)
    right_leg_theta = torch.acos(right_leg_dot)
    right_leg_theta = torch.where(right_knee_local_dir0[..., 1] >= 0, right_leg_theta, -right_leg_theta)
    right_leg_q = quat_from_angle_axis(right_leg_theta, right_knee_local_dir.unsqueeze(0))
    right_hip_rot = quat_mul(right_hip_rot, right_leg_q)
    
    # left leg
    left_thigh_pos = motion.global_translation[..., left_thigh_id, :]
    left_shin_pos = motion.global_translation[..., left_shin_id, :]
    left_foot_pos = motion.global_translation[..., left_foot_id, :]
    left_hip_rot = motion.local_rotation[..., left_thigh_id, :]
    left_knee_rot = motion.local_rotation[..., left_shin_id, :]
    
    left_leg_delta0 = left_thigh_pos - left_shin_pos
    left_leg_delta1 = left_foot_pos - left_shin_pos
    left_leg_delta0 = left_leg_delta0 / torch.norm(left_leg_delta0, dim=-1, keepdim=True)
    left_leg_delta1 = left_leg_delta1 / torch.norm(left_leg_delta1, dim=-1, keepdim=True)
    left_knee_dot = torch.sum(-left_leg_delta0 * left_leg_delta1, dim=-1)
    left_knee_dot = torch.clamp(left_knee_dot, -1.0, 1.0)
    left_knee_theta = torch.acos(left_knee_dot)
    left_knee_q = quat_from_angle_axis(torch.abs(left_knee_theta), torch.tensor(np.array([[0.0, 1.0, 0.0]]), 
                                        device=device, dtype=torch.float32))
    
    left_knee_local_dir = motion.skeleton_tree.local_translation[left_foot_id]
    left_knee_local_dir = left_knee_local_dir / torch.norm(left_knee_local_dir)
    left_knee_local_dir_tile = torch.tile(left_knee_local_dir.unsqueeze(0), [left_knee_rot.shape[0], 1])
    left_knee_local_dir0 = quat_rotate(left_knee_rot, left_knee_local_dir_tile)
    left_knee_local_dir1 = quat_rotate(left_knee_q, left_knee_local_dir_tile)
    left_leg_dot = torch.sum(left_knee_local_dir0 * left_knee_local_dir1, dim=-1)
    left_leg_dot = torch.clamp(left_leg_dot, -1.0, 1.0)
    left_leg_theta = torch.acos(left_leg_dot)
    left_leg_theta = torch.where(left_knee_local_dir0[..., 1] >= 0, left_leg_theta, -left_leg_theta)
    left_leg_q = quat_from_angle_axis(left_leg_theta, left_knee_local_dir.unsqueeze(0))
    left_hip_rot = quat_mul(left_hip_rot, left_leg_q)
    

    new_local_rotation = motion.local_rotation.clone()
    new_local_rotation[..., right_upper_arm_id, :] = right_shoulder_rot
    new_local_rotation[..., right_lower_arm_id, :] = right_elbow_q
    new_local_rotation[..., left_upper_arm_id, :] = left_shoulder_rot
    new_local_rotation[..., left_lower_arm_id, :] = left_elbow_q
    
    new_local_rotation[..., right_thigh_id, :] = right_hip_rot
    new_local_rotation[..., right_shin_id, :] = right_knee_q
    new_local_rotation[..., left_thigh_id, :] = left_hip_rot
    new_local_rotation[..., left_shin_id, :] = left_knee_q
    
    new_local_rotation[..., left_hand_id, :] = quat_identity([1])
    new_local_rotation[..., right_hand_id, :] = quat_identity([1])

    new_sk_state = SkeletonState.from_rotation_and_root_translation(motion.skeleton_tree, new_local_rotation, motion.root_translation, is_local=True)
    new_motion = SkeletonMotion.from_skeleton_state(new_sk_state, fps=motion.fps)
    
    return new_motion


def retarget(source_motion, src_tpose_fromM, src_tpose_path, target_motion_file, target_tpose_path, retarget_data, visualize, src_type):
    # load and visualize t-pose files
    target_tpose = SkeletonState.from_file(target_tpose_path)
    if visualize:
        plot_skeleton_state(target_tpose)

    scale, rotation = calc_adaptions_from_tpose(target_tpose, src_tpose_fromM, src_type)
    src_tpose = SkeletonState.from_file(src_tpose_path)
    if visualize:
        plot_skeleton_state(src_tpose_fromM)

    # load and visualize source motion sequence
    if visualize:
        plot_skeleton_motion_interactive(source_motion)

    # parse data from retarget config
    rotation_to_target_skeleton = torch.tensor(retarget_data["rotation"])

    # run retargeting
    target_motion = source_motion.retarget_to_by_tpose(
        joint_mapping=retarget_data["joint_mapping"],
        source_tpose=src_tpose,
        target_tpose=target_tpose,
        rotation_to_target_skeleton=rotation_to_target_skeleton,
        scale_to_target_skeleton=scale
    )

    # keep frames between [trim_frame_beg, trim_frame_end - 1]
    frame_beg = retarget_data["trim_frame_beg"]
    frame_end = retarget_data["trim_frame_end"]
    if (frame_beg == -1):
        frame_beg = 0

    if (frame_end == -1):
        frame_end = target_motion.local_rotation.shape[0]

    local_rotation = target_motion.local_rotation
    root_translation = target_motion.root_translation
    local_rotation = local_rotation[frame_beg:frame_end, ...]
    root_translation = root_translation[frame_beg:frame_end, ...]

    new_sk_state = SkeletonState.from_rotation_and_root_translation(target_motion.skeleton_tree, local_rotation,
                                                                    root_translation, is_local=True)
    target_motion = SkeletonMotion.from_skeleton_state(new_sk_state, fps=target_motion.fps)

    # need to convert some joints from 3D to 1D (e.g. elbows and knees)
    target_motion = project_joints(target_motion)

    # move the root so that the feet are on the ground
    local_rotation = target_motion.local_rotation
    root_translation = target_motion.root_translation
    tar_global_pos = target_motion.global_translation
    min_h = torch.min(tar_global_pos[..., 2])
    root_translation[:, 2] += -min_h

    # adjust the height of the root to avoid ground penetration
    root_height_offset = retarget_data["root_height_offset"]
    root_translation[:, 2] += root_height_offset

    new_sk_state = SkeletonState.from_rotation_and_root_translation(target_motion.skeleton_tree, local_rotation,
                                                                    root_translation, is_local=True)
    target_motion = SkeletonMotion.from_skeleton_state(new_sk_state, fps=target_motion.fps)

    # save retargeted motion
    target_motion.to_file(target_motion_file)

    # visualize retargeted motion
    if visualize:
        plot_skeleton_motion_interactive(target_motion)

    return


def retarget_motions_list(src_motions_path, src_motion_file_list, src_motion_list, src_tpose_list, src_tpose_path, target_motion_path,
                          target_tpose_paths, retarget_data, visualize, src_type):
    if len(src_motion_file_list) == 0:
        return
    for i in range(len(src_motion_file_list)):
        m_file = src_motion_file_list[i]
        src_tpose = src_tpose_list[i]
        src_motion = src_motion_list[i]

        dst_files, dirs_out = get_dst_file_and_folders(src_motions_path, target_motion_path, target_tpose_paths,
                                                       m_file, 4, ".npy")

        for j, d_out in enumerate(dirs_out):
            if not os.path.exists(d_out):
                os.makedirs(d_out)
            dst_file = dst_files[j]
            target_tpose_path = target_tpose_paths[j]
            retarget(src_motion, src_tpose, src_tpose_path, dst_file, target_tpose_path, retarget_data, visualize, src_type)


def load_motions_info(motions_files_list, visualize, src_fps, src_type):
    if len(motions_files_list) == 0:
        return [], []
    else:
        motion_list = []
        tpose_list = []
        for m_file in motions_files_list:
            assert m_file.endswith(".fbx"), f"The source motion files must be an fbx format, but {m_file} was given"
            motion = SkeletonMotion.from_fbx(
                fbx_file_path=m_file,
                root_joint="Hips",
                fps=src_fps)
            motion_list.append(motion)
            tpose = estimate_tpose_from_motion(motion, src_type)
            tpose_list.append(tpose)

            if visualize:
                plot_skeleton_motion_interactive(motion)
        return motion_list, tpose_list


def get_dst_file_and_folders(input_folder, output_folder, target_tpose_paths, filename_path, len_input_ext=0, new_ext=""):
    rest_path = filename_path[len(input_folder):]
    if rest_path.startswith("/"):
        rest_path = rest_path[1:]
    dst_files = []
    dirs_out = []
    for target_tpose_path in target_tpose_paths:
        target_tpose_name = os.path.basename(target_tpose_path)[:-4]
        # regex for matching A.AA for height in target in name
        reg_pattern = "_[0-9][0-9][0-9]_"
        match = re.search(reg_pattern, target_tpose_name)
        if match is None:
            #use the whole name for the subfolder
            subfolder_name = target_tpose_name
        else:
            subfolder_name = match.group()
            #get rid of "_"
            subfolder_name = subfolder_name[1:-1]

        dst_file = os.path.join(output_folder, subfolder_name, rest_path)[:-len_input_ext] + new_ext
        dirs_out_path = os.path.dirname(dst_file)
        dst_files.append(dst_file)
        dirs_out.append(dirs_out_path)
    return dst_files, dirs_out


def get_motions_list(path):
    fbx_files_list = glob.glob(glob.escape(path) + "/**/*.fbx", recursive=True)
    return fbx_files_list


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_motions_path', type=str,
                        help='path to the source motions. Either a folder for many motions or a the name of a file for a single motion',
                        required=True)
    parser.add_argument('--src_type',
                        type=str,
                        choices=['cmu', 'lafan', 'zeggs', 'bandai_namco'],
                        help='The type corresponds to the database and its respective model from the source motions',
                        required=True)
    parser.add_argument('--target_tpose_paths', nargs='+', default=[],
                        help='tpose file or files for the skeleton into which we reterget the motions',
                        required=True)
    parser.add_argument('--target_motions_path', type=str,
                        help='path to the retargeted motions. Either a folder or a the name of a file for a single motion',
                        required=True)
    parser.add_argument('--visualize', action='store_true', help='Visualize the different steps')
    args = parser.parse_args()

    assert os.path.exists(args.src_motions_path), "The source motion path does not exist."
    for target_path in args.target_tpose_paths:
        assert os.path.exists(target_path)
    if args.src_type == 'cmu':
        src_tpose_path = 'data/cmu_tpose.npy'
        retarget_data_path = 'data/configs/retarget_cmu_to_amp_general.json'
        src_fps = 60
    elif args.src_type == 'sfu':
        src_tpose_path = 'data/sfu_tpose.npy'
        retarget_data_path = 'data/configs/retarget_sfu_to_amp_general.json'
        src_fps = 60
    elif args.src_type == 'lafan':
        src_tpose_path = 'data/lafan_tpose.npy'
        retarget_data_path = 'data/configs/retarget_lafan_to_amp_general.json'
        src_fps = 30
    elif args.src_type == 'zeggs':
        src_tpose_path = 'data/zeggs_tpose.npy'
        retarget_data_path = 'data/configs/retarget_zeggs_to_amp_general.json'
        src_fps = 120
    elif args.src_type == 'bandai_namco':
        src_tpose_path = 'data/bandai_namco_tpose.npy'
        retarget_data_path = 'data/configs/retarget_bandai_namco_to_amp_general.json'
        src_fps = 30

    assert os.path.exists(src_tpose_path), f"The source tpose file {src_tpose_path} does not exist."
    assert os.path.exists(retarget_data_path), f"The retarget data file {retarget_data_path} does not exist."

    src_motion_file_list = get_motions_list(args.src_motions_path)
    src_motion_list, src_tpose_list = load_motions_info(src_motion_file_list, args.visualize, src_fps,
                                                        args.src_type)
    with open(retarget_data_path) as f:
        retarget_data = json.load(f)

    retarget_motions_list(args.src_motions_path, src_motion_file_list, src_motion_list, src_tpose_list,
                          src_tpose_path, args.target_motions_path, args.target_tpose_paths, retarget_data,
                          args.visualize, args.src_type)


if __name__ == '__main__':
    main()