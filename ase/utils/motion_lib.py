# Copyright (c) 2018-2022, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import numpy as np
import os
import yaml

from isaacgym.torch_utils import *
from poselib.poselib.skeleton.skeleton3d import SkeletonMotion
from poselib.poselib.core.rotation3d import *


from utils import torch_utils

import torch
import copy

USE_CACHE = True
print("MOVING MOTION DATA TO GPU, USING CACHE:", USE_CACHE)

if not USE_CACHE:
    old_numpy = torch.Tensor.numpy
    class Patch:
        def numpy(self):
            if self.is_cuda:
                return self.to("cpu").numpy()
            else:
                return old_numpy(self)

    torch.Tensor.numpy = Patch.numpy

class DeviceCache:
    def __init__(self, obj, device):
        self.obj = obj
        self.device = device

        keys = dir(obj)
        num_added = 0
        for k in keys:
            try:
                out = getattr(obj, k)
            except:
                #according tO ASE repo this is ok
                #print("Error for key=", k)
                continue

            if isinstance(out, torch.Tensor):
                if out.is_floating_point():
                    out = out.to(self.device, dtype=torch.float32)
                else:
                    out.to(self.device)
                setattr(self, k, out)  
                num_added += 1
            elif isinstance(out, np.ndarray):
                out = torch.tensor(out)
                if out.is_floating_point():
                    out = out.to(self.device, dtype=torch.float32)
                else:
                    out.to(self.device)
                setattr(self, k, out)
                num_added += 1

        #print("Total added", num_added)

    def __getattr__(self, string):
        out = getattr(self.obj, string)
        return out


class MotionLib():
    def __init__(self, motion_file, dof_body_ids, dof_offsets,
                 key_body_ids, device, verbose=True, already_read_file=False):
        self.verbose=verbose
        self._dof_body_ids = dof_body_ids
        self._dof_offsets = dof_offsets
        self._num_dof = dof_offsets[-1]
        self._key_body_ids = torch.tensor(key_body_ids, device=device)
        self._device = device
        self._already_read_file = already_read_file
        self._load_motions(motion_file)

        motions = self._motions
        self.gts = torch.cat([m.global_translation for m in motions], dim=0).float()
        self.grs = torch.cat([m.global_rotation for m in motions], dim=0).float()
        self.gvs = torch.cat([m.global_velocity for m in motions], dim=0).float()
        self.gavs = torch.cat([m.global_angular_velocity for m in motions], dim=0).float()
        self.lrs = torch.cat([m.local_rotation for m in motions], dim=0).float()
        self.grvs = torch.cat([m.global_root_velocity for m in motions], dim=0).float()
        self.gravs = torch.cat([m.global_root_angular_velocity for m in motions], dim=0).float()
        self.dvs = torch.cat([m.dof_vels for m in motions], dim=0).float()

        lengths = self._motion_num_frames
        lengths_shifted = lengths.roll(1)
        lengths_shifted[0] = 0
        self.length_starts = lengths_shifted.cumsum(0)

        self.motion_ids = torch.arange(len(self._motions), dtype=torch.long, device=self._device)

        return

    def num_motions(self):
        return len(self._motions)

    def get_total_length(self):
        return sum(self._motion_lengths)

    def get_motion(self, motion_id):
        return self._motions[motion_id]

    def get_motion_name(self, motion_id):
        return self._motion_names[motion_id]

    def sample_motions(self, n):
        motion_ids = torch.multinomial(self._motion_weights, num_samples=n, replacement=True)

        # m = self.num_motions()
        # motion_ids = np.random.choice(m, size=n, replace=True, p=self._motion_weights)
        # motion_ids = torch.tensor(motion_ids, device=self._device, dtype=torch.long)
        return motion_ids

    def sample_time(self, motion_ids, truncate_time=None):
        n = len(motion_ids)
        phase = torch.rand(motion_ids.shape, device=self._device)
        
        motion_len = self._motion_lengths[motion_ids]
        if (truncate_time is not None):
            assert(truncate_time >= 0.0)
            motion_len -= truncate_time

        motion_time = phase * motion_len
        return motion_time

    def get_motion_length(self, motion_ids):
        return self._motion_lengths[motion_ids]

    def get_motion_state(self, motion_ids, motion_times):
        n = len(motion_ids)
        num_bodies = self._get_num_bodies()
        num_key_bodies = self._key_body_ids.shape[0]

        motion_len = self._motion_lengths[motion_ids]
        num_frames = self._motion_num_frames[motion_ids]
        dt = self._motion_dt[motion_ids]

        frame_idx0, frame_idx1, blend = self._calc_frame_blend(motion_times, motion_len, num_frames, dt)

        f0l = frame_idx0 + self.length_starts[motion_ids]
        f1l = frame_idx1 + self.length_starts[motion_ids]

        root_pos0 = self.gts[f0l, 0]
        root_pos1 = self.gts[f1l, 0]

        root_rot0 = self.grs[f0l, 0]
        root_rot1 = self.grs[f1l, 0]

        local_rot0 = self.lrs[f0l]
        local_rot1 = self.lrs[f1l]

        root_vel = self.grvs[f0l]

        root_ang_vel = self.gravs[f0l]
        
        key_pos0 = self.gts[f0l.unsqueeze(-1), self._key_body_ids.unsqueeze(0)]
        key_pos1 = self.gts[f1l.unsqueeze(-1), self._key_body_ids.unsqueeze(0)]

        dof_vel = self.dvs[f0l]

        vals = [root_pos0, root_pos1, local_rot0, local_rot1, root_vel, root_ang_vel, key_pos0, key_pos1]
        for v in vals:
            assert v.dtype != torch.float64


        blend = blend.unsqueeze(-1)

        root_pos = (1.0 - blend) * root_pos0 + blend * root_pos1

        root_rot = torch_utils.slerp(root_rot0, root_rot1, blend)

        blend_exp = blend.unsqueeze(-1)
        key_pos = (1.0 - blend_exp) * key_pos0 + blend_exp * key_pos1
        
        local_rot = torch_utils.slerp(local_rot0, local_rot1, torch.unsqueeze(blend, axis=-1))
        dof_pos = self._local_rotation_to_dof(local_rot)

        return root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel, key_pos

    def get_rb_state(self, motion_ids, motion_times):
        n = len(motion_ids)
        num_bodies = self._get_num_bodies()
        num_key_bodies = self._key_body_ids.shape[0]

        motion_len = self._motion_lengths[motion_ids]
        num_frames = self._motion_num_frames[motion_ids]
        dt = self._motion_dt[motion_ids]

        frame_idx0, frame_idx1, blend = self._calc_frame_blend(motion_times, motion_len, num_frames, dt)

        f0l = frame_idx0 + self.length_starts[motion_ids]
        f1l = frame_idx1 + self.length_starts[motion_ids]

        pos0 = self.gts[f0l]
        pos1 = self.gts[f1l]

        rot0 = self.grs[f0l]
        rot1 = self.grs[f1l]

        local_rot0 = self.lrs[f0l]
        local_rot1 = self.lrs[f1l]

        vel = self.gvs[f0l]


        dof_vel = self.dvs[f0l]

        vals = [pos0, pos1, rot0, rot1]
        for v in vals:
            assert v.dtype != torch.float64

        blend = blend.unsqueeze(-1)
        blend = blend.unsqueeze(-1)

        pos = (1.0 - blend) * pos0 + blend * pos1

        rot = torch_utils.slerp(rot0, rot1, blend)

        local_rot = torch_utils.slerp(local_rot0, local_rot1, blend)
        dof_pos = self._local_rotation_to_dof(local_rot)

        ang_vel = self.gavs[f0l]

        return pos, rot, vel, dof_pos, dof_vel, ang_vel

    
    def _load_motions(self, motion_file):
        self._motions = []
        self._motion_lengths = []
        self._motion_weights = []
        self._motion_fps = []
        self._motion_dt = []
        self._motion_num_frames = []
        self._motion_files = []
        self._motion_names = []

        total_len = 0.0

        motion_files, motion_weights = self._fetch_motion_files(motion_file)
        num_motion_files = len(motion_files)
        for f in range(num_motion_files):
            curr_file = motion_files[f]
            if self.verbose:
                print("Loading {:d}/{:d} motion files: {:s}".format(f + 1, num_motion_files, curr_file))
            curr_motion = SkeletonMotion.from_file(curr_file)

            motion_fps = curr_motion.fps
            curr_dt = 1.0 / motion_fps

            num_frames = curr_motion.tensor.shape[0]
            curr_len = 1.0 / motion_fps * (num_frames - 1)

            self._motion_fps.append(motion_fps)
            self._motion_dt.append(curr_dt)
            self._motion_num_frames.append(num_frames)
 
            curr_dof_vels = self._compute_motion_dof_vels(curr_motion)
            curr_motion.dof_vels = curr_dof_vels

            # Moving motion tensors to the GPU
            if USE_CACHE:
                curr_motion = DeviceCache(curr_motion, self._device)                
            else:
                curr_motion.tensor = curr_motion.tensor.to(self._device)
                curr_motion._skeleton_tree._parent_indices = curr_motion._skeleton_tree._parent_indices.to(self._device)
                curr_motion._skeleton_tree._local_translation = curr_motion._skeleton_tree._local_translation.to(self._device)
                curr_motion._rotation = curr_motion._rotation.to(self._device)

            self._motions.append(curr_motion)
            self._motion_lengths.append(curr_len)
            
            curr_weight = motion_weights[f]
            self._motion_weights.append(curr_weight)
            self._motion_files.append(curr_file)
            name_parts = curr_file.split(os.sep)
            m_name = name_parts[-1]
            self._motion_names.append(m_name)

        self._motion_lengths = torch.tensor(self._motion_lengths, device=self._device, dtype=torch.float32)

        self._motion_weights = torch.tensor(self._motion_weights, dtype=torch.float32, device=self._device)
        self._motion_weights /= self._motion_weights.sum()

        self._motion_fps = torch.tensor(self._motion_fps, device=self._device, dtype=torch.float32)
        self._motion_dt = torch.tensor(self._motion_dt, device=self._device, dtype=torch.float32)
        self._motion_num_frames = torch.tensor(self._motion_num_frames, device=self._device)


        num_motions = self.num_motions()
        total_len = self.get_total_length()

        if self.verbose:
            print("Loaded {:d} motions with a total length of {:.3f}s.".format(num_motions, total_len))

        return

    def _fetch_motion_files(self, motion_file):
        if not self._already_read_file:
            ext = os.path.splitext(motion_file)[1]
        if self._already_read_file or (ext == ".yaml"):
            motion_files = []
            motion_weights = []
            if not self._already_read_file:
                dir_name = os.path.dirname(motion_file)
                with open(os.path.join(os.getcwd(), motion_file), 'r') as f:
                    motion_config = yaml.load(f, Loader=yaml.SafeLoader)
            else:
                motion_config = motion_file
                assert "wd_path" in motion_config.keys()
                assert motion_config['wd_path']

            motion_list = motion_config['motions']
            if "wd_path" in motion_config.keys():
                from_working_dir = motion_config['wd_path']
            else:
                from_working_dir = False
            for motion_entry in motion_list:
                curr_file = motion_entry['file']
                curr_weight = motion_entry['weight']
                assert(curr_weight >= 0)

                if from_working_dir:
                    curr_file = curr_file
                else:
                    curr_file = os.path.join(dir_name, curr_file)
                motion_weights.append(curr_weight)
                motion_files.append(curr_file)
        else:
            motion_files = [motion_file]
            motion_weights = [1.0]

        return motion_files, motion_weights

    def _calc_frame_blend(self, time, len, num_frames, dt):

        phase = time / len
        phase = torch.clip(phase, 0.0, 1.0)

        frame_idx0 = (phase * (num_frames - 1)).long()
        frame_idx1 = torch.min(frame_idx0 + 1, num_frames - 1)
        blend = (time - frame_idx0 * dt) / dt

        return frame_idx0, frame_idx1, blend

    def _get_num_bodies(self):
        motion = self.get_motion(0)
        num_bodies = motion.num_joints
        return num_bodies

    def _compute_motion_dof_vels(self, motion):
        num_frames = motion.tensor.shape[0]
        dt = 1.0 / motion.fps
        dof_vels = []

        for f in range(num_frames - 1):
            local_rot0 = motion.local_rotation[f]
            local_rot1 = motion.local_rotation[f + 1]
            frame_dof_vel = self._local_rotation_to_dof_vel(local_rot0, local_rot1, dt)
            frame_dof_vel = frame_dof_vel
            dof_vels.append(frame_dof_vel)
        
        dof_vels.append(dof_vels[-1])
        dof_vels = torch.stack(dof_vels, dim=0)

        return dof_vels
    
    def _local_rotation_to_dof(self, local_rot):
        body_ids = self._dof_body_ids
        dof_offsets = self._dof_offsets

        n = local_rot.shape[0]
        dof_pos = torch.zeros((n, self._num_dof), dtype=torch.float, device=self._device)

        for j in range(len(body_ids)):
            body_id = body_ids[j]
            joint_offset = dof_offsets[j]
            joint_size = dof_offsets[j + 1] - joint_offset

            if (joint_size == 3):
                joint_q = local_rot[:, body_id]
                joint_exp_map = torch_utils.quat_to_exp_map(joint_q)
                dof_pos[:, joint_offset:(joint_offset + joint_size)] = joint_exp_map
            elif (joint_size == 1):
                joint_q = local_rot[:, body_id]
                joint_theta, joint_axis = torch_utils.quat_to_angle_axis(joint_q)
                joint_theta = joint_theta * joint_axis[..., 1] # assume joint is always along y axis

                joint_theta = normalize_angle(joint_theta)
                dof_pos[:, joint_offset] = joint_theta

            else:
                print("Unsupported joint type")
                assert(False)

        return dof_pos

    def _local_rotation_to_dof_vel(self, local_rot0, local_rot1, dt):
        body_ids = self._dof_body_ids
        dof_offsets = self._dof_offsets

        dof_vel = torch.zeros([self._num_dof], device=self._device)

        diff_quat_data = quat_mul_norm(quat_inverse(local_rot0), local_rot1)
        diff_angle, diff_axis = quat_angle_axis(diff_quat_data)
        local_vel = diff_axis * diff_angle.unsqueeze(-1) / dt
        local_vel = local_vel

        for j in range(len(body_ids)):
            body_id = body_ids[j]
            joint_offset = dof_offsets[j]
            joint_size = dof_offsets[j + 1] - joint_offset

            if (joint_size == 3):
                joint_vel = local_vel[body_id]
                dof_vel[joint_offset:(joint_offset + joint_size)] = joint_vel

            elif (joint_size == 1):
                assert(joint_size == 1)
                joint_vel = local_vel[body_id]
                dof_vel[joint_offset] = joint_vel[1] # assume joint is always along y axis

            else:
                print("Unsupported joint type")
                assert(False)

        return dof_vel


# A Wrapper to load motion files in different motion files
# Currently used to load different heights
class MultipleMotionLib():
    def __init__(self, motion_file, dof_body_ids, dof_offsets, key_body_ids, device, file_folder_replace,
                 file_folder_replacements, keys_list):
        assert motion_file.endswith(".yaml"), "When using wrapper provide in the yaml all the required motions"


        single_file = False
        if motion_file.endswith(".yaml"):
            with open(os.path.join(os.getcwd(), motion_file), 'r') as f:
                motions_dict_list = yaml.load(f, Loader=yaml.SafeLoader)
        else:
            single_file = True

        self.motion_libs = []
        self.keys = []
        self.keys_to_ids = {}
        for i, repl in enumerate(file_folder_replacements):
            key = keys_list[i]
            self.keys.append(key)
            self.keys_to_ids[key] = i
            if single_file:
                repl_mfile = motion_file.replace(file_folder_replace, repl)
                self.motion_libs.append(MotionLib(repl_mfile, dof_body_ids, dof_offsets,
                                                  key_body_ids, device))
            else:
                motions_dict_list_copy = copy.deepcopy(motions_dict_list)
                for j in range(len(motions_dict_list_copy["motions"])):
                    mfname = motions_dict_list_copy["motions"][j]["file"]
                    mfname = mfname.replace(file_folder_replace, repl)
                    motions_dict_list_copy["motions"][j]["file"] = mfname
                self.motion_libs.append(MotionLib(motions_dict_list_copy, dof_body_ids, dof_offsets,
                     key_body_ids, device, already_read_file=True))

    def num_motions(self):
        return self.motion_libs[0].num_motions()
    
    def get_motion_name(self, motion_id):
        return self.motion_libs[0].get_motion_name(motion_id)

    def sample_motions(self, n):
        return self.motion_libs[0].sample_motions(n)

    def sample_time(self, motion_ids, truncate_time=None):
        return self.motion_libs[0].sample_time(motion_ids, truncate_time=truncate_time)

    def get_motion_length(self, motion_ids):
        return self.motion_libs[0].get_motion_length(motion_ids)

    def get_motion_state(self, motion_ids, motion_times, motion_heights):
        #first get sample from first motion_lib so that shape is correct
        ref_root_pos, ref_root_rot, ref_dof_pos, ref_root_vel, ref_root_ang_vel, ref_dof_vel, ref_key_pos = self.motion_libs[0].get_motion_state(motion_ids, motion_times)
        ref_root_pos[...] = 0.0
        ref_root_rot[...] = 0.0
        ref_dof_pos[...] = 0.0
        ref_root_vel[...] = 0.0
        ref_root_ang_vel[...] = 0.0
        ref_dof_vel[...] = 0.0
        ref_key_pos[...] = 0.0

        all_check_tensor = torch.tensor([False]*motion_heights.shape[0], device=ref_dof_pos.device)

        for key in self.keys:
            # key to height
            motions_h_mask = motion_heights == key

            root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel, key_pos = self.motion_libs[self.keys_to_ids[key]].get_motion_state(motion_ids, motion_times)
            ref_root_pos[motions_h_mask] = root_pos[motions_h_mask]
            ref_root_rot[motions_h_mask] = root_rot[motions_h_mask]
            ref_dof_pos[motions_h_mask] = dof_pos[motions_h_mask]
            ref_root_vel[motions_h_mask] = root_vel[motions_h_mask]
            ref_root_ang_vel[motions_h_mask] = root_ang_vel[motions_h_mask]
            ref_dof_vel[motions_h_mask] = dof_vel[motions_h_mask]
            ref_key_pos[motions_h_mask] = key_pos[motions_h_mask]

            all_check_tensor = torch.logical_or(all_check_tensor, motions_h_mask)

        # all given heights should have been matched
        assert torch.all(all_check_tensor).item()

        return ref_root_pos, ref_root_rot, ref_dof_pos, ref_root_vel, ref_root_ang_vel, ref_dof_vel, ref_key_pos

    def get_rb_state(self, motion_ids, motion_times, motion_heights):
        ref_pos, ref_rot, ref_vel, ref_dof_pos, ref_dof_vel, ref_ang_vel = self.motion_libs[0].get_rb_state(motion_ids, motion_times)
        ref_pos[...] = 0.0
        ref_rot[...] = 0.0
        ref_vel[...] = 0.0
        ref_dof_pos[...] = 0.0
        ref_dof_vel[...] = 0.0
        ref_ang_vel[...] = 0.0

        all_check_tensor = torch.tensor([False] * motion_heights.shape[0], device=ref_pos.device)

        for key in self.keys:
            # key to height
            motions_h_mask = motion_heights == key

            pos, rot, vel, dof_pos, dof_vel, ang_vel = self.motion_libs[self.keys_to_ids[key]].get_rb_state(motion_ids, motion_times)
            ref_pos[motions_h_mask] = pos[motions_h_mask]
            ref_rot[motions_h_mask] = rot[motions_h_mask]
            ref_vel[motions_h_mask] = vel[motions_h_mask]
            ref_dof_pos[motions_h_mask] = dof_pos[motions_h_mask]
            ref_dof_vel[motions_h_mask] = dof_vel[motions_h_mask]
            ref_ang_vel[motions_h_mask] = ang_vel[motions_h_mask]

            all_check_tensor = torch.logical_or(all_check_tensor, motions_h_mask)

            # all given heights should have been matched
        assert torch.all(all_check_tensor).item()
        return ref_pos, ref_rot, ref_vel, ref_dof_pos, ref_dof_vel, ref_ang_vel







