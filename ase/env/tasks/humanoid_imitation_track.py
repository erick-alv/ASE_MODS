import torch

from env.tasks.humanoid import compute_humanoid_reset
from env.tasks.humanoid_motion_load_and_reset import HumanoidMotionAndReset
from isaacgym.torch_utils import *
from isaacgym import gymapi, gymutil, gymtorch
from real_time.utils import to_rotations_tensor, to_positions_tensor, reorder_device
from utils import env_obs_util, env_rew_util
import math
import time


class HumanoidImitationTrack(HumanoidMotionAndReset):
    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        # todo del once not more used
        self.pre_obs = False

        self.dof_rot_rew = cfg["env"].get("dof_rot_rew", False)
        self.include_global_obs = cfg["env"].get("include_global_obs", False)
        super().__init__(cfg=cfg,
                         sim_params=sim_params,
                         physics_engine=physics_engine,
                         device_type=device_type,
                         device_id=device_id,
                         headless=headless)

        if not cfg["env"]["real_time"]:
            num_motions = self._motion_lib.num_motions()
            self._motion_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
            self._motion_ids = torch.remainder(self._motion_ids, num_motions)
            self._motions_start_time = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)

        self.prev_feet_contact_forces = torch.clone(self.feet_contact_forces)
        #  read parameters specific for imitating the movements based on track readings
        self.num_steps_track_info = self.cfg["env"]["imitParams"]["num_steps_track_info"]
        self.joint_friction = self.cfg["env"]["imitParams"]["joint_friction"]
        #  params for imitation reward
        self.reward_ws = {}
        self.reward_ks = {}
        self.reward_ws["w_dof_pos"] = self.cfg["env"]["imitParams"]["w_dof_pos"]
        self.reward_ws["w_dof_vel"] = self.cfg["env"]["imitParams"]["w_dof_vel"]
        self.reward_ws["w_pos"] = self.cfg["env"]["imitParams"]["w_pos"]
        self.reward_ws["w_vel"] = self.cfg["env"]["imitParams"]["w_vel"]
        self.reward_ws["w_force"] = self.cfg["env"]["imitParams"]["w_force"]
        self.reward_ks["k_dof_pos"] = self.cfg["env"]["imitParams"]["k_dof_pos"]
        self.reward_ks["k_dof_vel"] = self.cfg["env"]["imitParams"]["k_dof_vel"]
        self.reward_ks["k_pos"] = self.cfg["env"]["imitParams"]["k_pos"]
        self.reward_ks["k_vel"] = self.cfg["env"]["imitParams"]["k_vel"]
        self.reward_ks["k_force"] = self.cfg["env"]["imitParams"]["k_force"]
        #  extra values for rewards variations
        self.reward_ws["w_extra1"] = self.cfg["env"]["imitParams"]["w_extra1"]
        self.reward_ws["w_extra2"] = self.cfg["env"]["imitParams"]["w_extra2"]
        self.reward_ws["w_extra3"] = self.cfg["env"]["imitParams"]["w_extra3"]
        self.reward_ks["k_extra1"] = self.cfg["env"]["imitParams"]["k_extra1"]
        self.reward_ks["k_extra2"] = self.cfg["env"]["imitParams"]["k_extra2"]
        self.reward_ks["k_extra3"] = self.cfg["env"]["imitParams"]["k_extra3"]
        # for updating weights
        if "r_weight_update" in self.cfg["env"].keys():
            self.do_weight_update = True
            self.reward_updates_dict = self.cfg["env"]["r_weight_update"]
            self.during_reward_smooth = False
        else:
            self.do_weight_update = False

        if "penalty_update" in self.cfg["env"].keys():
            self.do_penalty_update = True
            self.penalty_updates_dict = self.cfg["env"]["penalty_update"]
            self.during_penalty_smooth = False
        else:
            self.do_penalty_update = False
        
        # for selecting reward function
        self.reward_type = self.cfg["env"]["reward_type"]
        # fall penalty; just used if reward function penalizes it
        self.fall_penalty = self.cfg["env"]["fall_penalty"]



        #TODO delete once debugged
        self.deb_sync = False
        self.deb_rew_print = 0
        #todo del
        self.deb2 = False





    @property
    def feet_contact_forces(self):
        #fcf = self.vec_sensor_tensor.view(self.num_envs, 2, 6)[:, :, :3] #the first 3 arguments correspond to the linear force
        a = self._contact_forces.view(self.num_envs, self.num_bodies, 3)[:, self._contact_feet_ids, :]
        return a

    # Overloaded methods from Humanoid

    def _setup_character_props(self, key_bodies):
        asset_file = self.cfg["env"]["asset"]["assetFileName"]
        num_key_bodies = len(key_bodies)
        device = self.cfg["args"].device + ':' + str(self.cfg["args"].device_id)
        self._rigid_body_track_indices = self.cfg["env"]["asset"]["trackIndices"]
        self._rigid_body_track_indices = torch.tensor(self._rigid_body_track_indices, dtype=torch.long, device=device)
        self._rigid_body_joints_indices = self.cfg["env"]["asset"]["jointsIndices"]
        self._rigid_body_joints_indices = torch.tensor(self._rigid_body_joints_indices, dtype=torch.long, device=device)
        track_bodies = self.cfg["env"]["asset"]["trackBodies"]
        self.num_track_dev = torch.tensor(len(track_bodies), dtype=torch.long, device=device)
        for i, track_name in enumerate(track_bodies):
            if track_name == "headset":
                track_headset_index = i
        self._track_headset_index = torch.tensor(track_headset_index, dtype=torch.long, device=device)


        num_steps_track_info = self.cfg["env"]["imitParams"]["num_steps_track_info"]

        supported_files = [ "mjcf/amp_humanoid_vrh_140.xml", "mjcf/amp_humanoid_vrh_152.xml", "mjcf/amp_humanoid_vrh_160.xml",
                            "mjcf/amp_humanoid_vrh_168.xml", "mjcf/amp_humanoid_vrh_180.xml", "mjcf/amp_humanoid_vrh_185.xml",
                            "mjcf/amp_humanoid_vrh_193.xml", "mjcf/amp_humanoid_vrh_207.xml", "mjcf/amp_humanoid_vrh_212.xml",
                            "mjcf/amp_humanoid_vrh_220.xml", "mjcf/amp_humanoid_vrh.xml",
                            ]
        supported_files_m2 = ["mjcf/amp_humanoid_vrhm2_140.xml", "mjcf/amp_humanoid_vrhm2_152.xml",
                              "mjcf/amp_humanoid_vrhm2_160.xml", "mjcf/amp_humanoid_vrhm2_168.xml", 
                              "mjcf/amp_humanoid_vrhm2_180.xml", "mjcf/amp_humanoid_vrhm2_185.xml",
                              "mjcf/amp_humanoid_vrhm2_193.xml", "mjcf/amp_humanoid_vrhm2_207.xml",
                              "mjcf/amp_humanoid_vrhm2_212.xml", "mjcf/amp_humanoid_vrhm2_220.xml"]

        all_supported_files = supported_files + supported_files_m2
        if (type(asset_file) is list and asset_file[0] in all_supported_files) or (asset_file in all_supported_files):
            if (type(asset_file) is list and asset_file[0] in supported_files) or (asset_file in supported_files):
                self._dof_body_ids = [1, 2, 4, 5, 8, 9, 12, 13, 14, 15, 16, 17]
            elif (type(asset_file) is list and asset_file[0] in supported_files_m2) or (asset_file in supported_files_m2):
                self._dof_body_ids = [2, 3, 5, 6, 9, 10, 13, 14, 15, 17, 18, 19]
                
            self._dof_offsets = [0, 3, 6, 9, 10, 13, 14, 17, 18, 21, 24, 25, 28]
            self._dof_obs_size = 72  # 12 bodies * 6; left from original code
            self._num_actions = 28
            # for estimating:
            # num_obs = (num_actions * 2) + #joints pieces * 15 + feet_contact forces + (#track pieces * 9 * number frames to take) + 2
            # + 1 for reset val
            # the last two are the observations for the height of motion that is being imitated and the height of the humanoid in the simulation
            # self._num_obs = 287 + 162 + 2
            self._num_obs = self._num_actions * 2 + self._rigid_body_joints_indices.size()[0] * 15 + 6 + \
                            self._rigid_body_track_indices.size()[0] * 9 * num_steps_track_info + 2 +1
            if self.pre_obs:
                self._num_obs -= 1

            if self.include_global_obs:
                # + 2 * 3 for global positions
                self._num_obs += 2 * 3

        else:
            print("Unsupported character config file: {}".format(asset_file))
            assert (False)

    def _create_envs(self, num_envs, spacing, num_per_row):
        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        use_multiple_heights = self.cfg["env"]["asset"]["multipleHeights"]
        asset_root = self.cfg["env"]["asset"]["assetRoot"]

        if use_multiple_heights:
            asset_file_list = self.cfg["env"]["asset"]["assetFileName"]
            asset_height_list = self.cfg["env"]["asset"]["assetHeight"]
            humanoid_asset_list = [self._load_humanoid_asset(asset_root=asset_root, asset_file=asset_file) for
                                   asset_file in asset_file_list]
            ref_humanoid_asset = humanoid_asset_list[0]
        else:
            asset_file = self.cfg["env"]["asset"]["assetFileName"]
            asset_height = self.cfg["env"]["asset"]["assetHeight"]
            humanoid_asset = self._load_humanoid_asset(asset_root, asset_file)
            ref_humanoid_asset = humanoid_asset

        actuator_props = self.gym.get_asset_actuator_properties(ref_humanoid_asset)
        motor_efforts = [prop.motor_effort for prop in actuator_props]

        self.max_motor_effort = max(motor_efforts)
        self.motor_efforts = to_torch(motor_efforts, device=self.device)

        self.torso_index = 0
        self.num_bodies = self.gym.get_asset_rigid_body_count(ref_humanoid_asset)
        self.num_dof = self.gym.get_asset_dof_count(ref_humanoid_asset)
        self.num_joints = self.gym.get_asset_joint_count(ref_humanoid_asset)

        self.humanoid_handles = []
        self.envs = []
        self.dof_limits_lower = []
        self.dof_limits_upper = []
        self.humanoid_heights = []
        self.imit_motion_heights = []

        self.mix_imit_heights = self.cfg["env"]["asset"]["mixImitHeights"]
        self.joint_friction = self.cfg["env"]["imitParams"]["joint_friction"]

        if self.cfg["env"]["real_time"]:
            while not (self.imitState.has_start_pose()):
                # print("Waiting for start pose")
                time.sleep(0.1)

            start_pose = self.imitState.get_start_pose()
            start_position_height = start_pose[0][0][2]

        for i in range(self.num_envs):
            # create env instance
            env_ptr = self.gym.create_env(self.sim, lower, upper, num_per_row)
            if use_multiple_heights:
                a_id = torch.randint(low=0, high=len(humanoid_asset_list), size=(1,))
                humanoid_asset = humanoid_asset_list[a_id.item()]
                height = asset_height_list[a_id.item()]
                self.humanoid_heights.append(height)

                #When doing real_time there should be
                if self.cfg["env"]["real_time"]:
                    self.imit_motion_heights.append(start_position_height)
                elif self.mix_imit_heights:
                    imit_h_id = torch.randint(low=0, high=len(humanoid_asset_list), size=(1,))
                    imit_height = asset_height_list[imit_h_id.item()]
                    self.imit_motion_heights.append(imit_height)
                else:
                    self.imit_motion_heights.append(height)

                self._build_env(i, env_ptr, humanoid_asset)
            else:
                self.humanoid_heights.append(asset_height)
                self.imit_motion_heights.append(asset_height)
                self._build_env(i, env_ptr, humanoid_asset)
            self.envs.append(env_ptr)

            dof_prop = self.gym.get_actor_dof_properties(self.envs[i], self.humanoid_handles[i])
            dof_prop['friction'][:] = self.joint_friction
            self.gym.set_actor_dof_properties(self.envs[i], self.humanoid_handles[i], dof_prop)

        dof_prop = self.gym.get_actor_dof_properties(self.envs[0], self.humanoid_handles[0])
        for j in range(self.num_dof):
            if dof_prop['lower'][j] > dof_prop['upper'][j]:
                self.dof_limits_lower.append(dof_prop['upper'][j])
                self.dof_limits_upper.append(dof_prop['lower'][j])
            else:
                self.dof_limits_lower.append(dof_prop['lower'][j])
                self.dof_limits_upper.append(dof_prop['upper'][j])

        self.dof_limits_lower = to_torch(self.dof_limits_lower, device=self.device)
        self.dof_limits_upper = to_torch(self.dof_limits_upper, device=self.device)
        self.humanoid_heights = to_torch(self.humanoid_heights, device=self.device)
        self.imit_motion_heights = to_torch(self.imit_motion_heights, device=self.device)

        if (self._pd_control):
            self._build_pd_action_offset_scale()

    def _asset_start_pose(self, env_id):
        start_pose = gymapi.Transform()
        asset_file = self.cfg["env"]["asset"]["assetFileName"]
        char_h = 0.6 * self.humanoid_heights[env_id]

        start_pose.p = gymapi.Vec3(*get_axis_params(char_h, self.up_axis_idx))
        start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
        return start_pose

    def update_rew_weights(self, epoch_num):
        if self.do_weight_update:
            if epoch_num in self.reward_updates_dict.keys() or self.during_reward_smooth:
                if epoch_num in self.reward_updates_dict.keys():
                    self.current_update_epoch_start = epoch_num
                    self.current_reward_update_dict = self.reward_updates_dict[epoch_num]
                    self.current_base_vals_dict = {}
                    for w_name in self.current_reward_update_dict["w_names"]:
                        self.current_base_vals_dict[w_name] = self.reward_ws[w_name]

                if "smooth_steps" in self.current_reward_update_dict.keys():
                    self.during_reward_smooth = True
                    smooth_steps = self.current_reward_update_dict["smooth_steps"]
                    prc_update = (epoch_num - self.current_update_epoch_start) / smooth_steps
                    
                    if epoch_num >= self.current_update_epoch_start + smooth_steps:
                        self.during_reward_smooth = False
                else:
                    self.during_reward_smooth = False
                    prc_update = 1.0


                #print("Updating reward weights")
                w_names_list = self.current_reward_update_dict["w_names"]
                w_vals_list = self.current_reward_update_dict["w_vals"]
                for i in range(len(w_names_list)):
                    w_name = w_names_list[i]
                    assert w_name in self.reward_ws.keys()
                    base_val = self.current_base_vals_dict[w_name]
                    new_val = base_val * (1.0 - prc_update) + prc_update * w_vals_list[i]
                    self.reward_ws[w_name] = new_val
                

        if self.do_penalty_update:
            if epoch_num in self.penalty_updates_dict.keys() or self.during_penalty_smooth:
                if epoch_num in self.penalty_updates_dict.keys():
                    self.current_penalty_update_epoch_start = epoch_num
                    self.current_penalty_update_dict = self.penalty_updates_dict[epoch_num]
                    self.current_base_penalty = self.fall_penalty

                if "smooth_steps" in self.current_penalty_update_dict.keys():
                    self.during_penalty_smooth = True
                    smooth_steps = self.current_penalty_update_dict["smooth_steps"]
                    prc_update = (epoch_num - self.current_penalty_update_epoch_start) / smooth_steps
                    if epoch_num >= self.current_penalty_update_epoch_start + smooth_steps:
                        self.during_penalty_smooth = False
                else:
                    self.during_penalty_smooth = False
                    prc_update = 1.0

                base_val = self.current_base_penalty
                new_val = base_val * (1.0 - prc_update) + prc_update * self.current_penalty_update_dict["val"]
                self.fall_penalty = new_val

    def _compute_reward(self, actions):
        if self.cfg["env"]["real_time"]:
            #We cannot measure the reward
            pass
        else:
            if self.use_multiple_heights:
                # We use the asset height in order to estimate correctly the error
                rb_pos_gt, rb_rot_gt, rb_vel_gt, \
                    dof_pos_gt, dof_vel_gt, _ = self._motion_lib.get_rb_state(self._motion_ids,
                                                                           self.progress_buf * self.dt + self._motions_start_time,
                                                                           self.humanoid_heights)
            else:
                rb_pos_gt, rb_rot_gt, rb_vel_gt, \
                    dof_pos_gt, dof_vel_gt, _ = self._motion_lib.get_rb_state(self._motion_ids,
                                                                           self.progress_buf * self.dt + self._motions_start_time)
            feet_contact_forces = self.feet_contact_forces
            prev_feet_contact_forces = self.prev_feet_contact_forces
            if self.reward_type == 0:
                use_penalty = False
            elif self.reward_type == 1:
                use_penalty = True
            else:
                raise Exception("not valid reward chosen")

            if self.dof_rot_rew:
                reward_fn = env_rew_util.compute_reward_dofr
            else:
                reward_fn = env_rew_util.compute_reward


            # if self.deb_sync:
            #     rb_pos, _, rb_vel, d_pos, d_vel, _ = self._motion_lib.get_rb_state(self._motion_ids,
            #                                                                self.progress_buf * self.dt + self._motions_start_time)
            # else:
            d_pos = self._dof_pos
            d_vel = self._dof_vel
            rb_pos = self._rigid_body_pos
            rb_vel = self._rigid_body_vel
            rb_rot = self._rigid_body_rot



            rew = reward_fn(
                dof_pos=d_pos, dof_pos_gt=dof_pos_gt,
                dof_vel=d_vel, dof_vel_gt=dof_vel_gt,
                rigid_body_pos=rb_pos, rigid_body_pos_gt=rb_pos_gt,
                rigid_body_vel=rb_vel, rigid_body_vel_gt=rb_vel_gt,
                rigid_body_rot=rb_rot, rigid_body_rot_gt=rb_rot_gt,
                rigid_body_joints_indices=self._rigid_body_joints_indices,
                feet_contact_forces=feet_contact_forces, prev_feet_contact_forces=prev_feet_contact_forces,
                feet_bodies_ids=self._contact_feet_ids,
                termination_heights=self._termination_heights, key_bodies_ids=self._key_body_ids,
                fall_penalty=self.fall_penalty,
                w_dof_pos=self.reward_ws["w_dof_pos"], w_dof_vel=self.reward_ws["w_dof_vel"],
                w_pos=self.reward_ws["w_pos"], w_vel=self.reward_ws["w_vel"], w_force=self.reward_ws["w_force"],
                k_dof_pos=self.reward_ks["k_dof_pos"], k_dof_vel=self.reward_ks["k_dof_vel"],
                k_pos=self.reward_ks["k_pos"], k_vel=self.reward_ks["k_vel"], k_force=self.reward_ks["k_force"],
                w_extra1=self.reward_ws["w_extra1"], w_extra2=self.reward_ws["w_extra2"],
                w_extra3=self.reward_ws["w_extra3"],
                k_extra1=self.reward_ks["k_extra1"], k_extra2=self.reward_ks["k_extra2"],
                k_extra3=self.reward_ks["k_extra3"],
                use_penalty=use_penalty
            )
            self.check_is_valid(rew)
            self.rew_buf[:] = rew
            if self.deb_sync:
                if self.deb_rew_print % 100 == 0:
                    print(rew)
                    self.deb_rew_print = 1
                else:
                    self.deb_rew_print+=1

    def _compute_reset(self):
        if self.cfg["env"]["real_time"] or self.cfg["env"]["test"]:
            #super()._compute_reset()
            pass
        else:

            motion_lengths = self._motion_lib.get_motion_length(self._motion_ids)
            motion_reset_buf, motion_terminate_buf = compute_ref_motion_reset(self.reset_buf, motion_lengths,
                                                                              self.progress_buf,
                                                                              self.dt,
                                                                              self._motions_start_time)

            humanoid_reset_buf, humanoid_terminate_buf = compute_humanoid_reset(self.reset_buf, self.progress_buf,
                                                                               self._contact_forces, self._contact_body_ids,
                                                                               self._rigid_body_pos,
                                                                               self.max_episode_length,
                                                                               self._enable_early_termination,
                                                                               self._termination_heights)
            r_or = torch.logical_or(motion_reset_buf, humanoid_reset_buf)
            self.reset_buf[:] = torch.where(r_or, torch.ones_like(motion_reset_buf), torch.zeros_like(motion_reset_buf))

            t_or = torch.logical_or(motion_terminate_buf, humanoid_terminate_buf)
            self._terminate_buf[:] = torch.where(t_or, torch.ones_like(motion_terminate_buf),
                                                 torch.zeros_like(motion_terminate_buf))
        return

    def _reset_env_tensors(self, env_ids):
        if self.cfg["env"]["real_time"]:
            #It is not necessary to reset in real time because we do not use them
            pass
        else:
            if len(self._reset_default_env_ids) > 0: #need to find new motion id for these envs
                new_motion_ids = self._motion_lib.sample_motions(len(self._reset_default_env_ids))
                self._motion_ids[self._reset_default_env_ids] = new_motion_ids
                self._motions_start_time[self._reset_default_env_ids] = 0
            if len(self._reset_ref_env_ids) > 0:#these already have a respective motion
                self._motion_ids[self._reset_ref_env_ids] = self._reset_ref_motion_ids
                self._motions_start_time[self._reset_ref_env_ids] = self._reset_ref_motion_times

        super()._reset_env_tensors(env_ids)

    def _compute_humanoid_obs(self, env_ids=None):
        rb_poses_gt_acc = []
        rb_rots_gt_acc = []
        if self.cfg["env"]["real_time"]:
            # wait until enough samples, allthough we could start here
            while not (self.imitState.is_ready()):
                print("waiting to read obs")
                time.sleep(0.1)
            imitState_buffer = self.imitState.get()
            rb_poses_gt_acc = to_positions_tensor(imitState_buffer)
            rb_rots_gt_acc = to_rotations_tensor(imitState_buffer)
            rb_poses_gt_acc = reorder_device(rb_poses_gt_acc,
                                             input_order=self.cfg["env"]["imitParams"]["rtBodyInputOrder"],
                                             training_order=self.cfg["env"]["asset"]["trackBodies"])
            rb_rots_gt_acc = reorder_device(rb_rots_gt_acc,
                                             input_order=self.cfg["env"]["imitParams"]["rtBodyInputOrder"],
                                             training_order=self.cfg["env"]["asset"]["trackBodies"])

            rb_poses_gt_acc = rb_poses_gt_acc.reshape((rb_poses_gt_acc.shape[0] * rb_poses_gt_acc.shape[1], -1))
            rb_rots_gt_acc = rb_rots_gt_acc.reshape((rb_rots_gt_acc.shape[0] * rb_rots_gt_acc.shape[1], -1))

            #make copies for each environment
            rb_poses_gt_acc = torch.stack([rb_poses_gt_acc] * self.num_envs, dim=0)
            rb_rots_gt_acc = torch.stack([rb_rots_gt_acc] * self.num_envs, dim=0)
            self.extras["track_pos_gt"] = rb_poses_gt_acc
            self.extras["track_rot_gt"] = rb_rots_gt_acc
        else:
            for i in range(self.num_steps_track_info):
                if self.use_multiple_heights:
                    # we use the imit motion height since that is the observation that we are giving to the agent
                    # independently of the real height of the humanoid. (Heights could be the same or not)
                    rb_pos_gt, rb_rot_gt, \
                        _, _, _, _ = self._motion_lib.get_rb_state(self._motion_ids,
                                                                (self.progress_buf + i + 1) * self.dt + self._motions_start_time,
                                                                self.imit_motion_heights
                                                                )
                else:
                    rb_pos_gt, rb_rot_gt, \
                        _, _, _, _ = self._motion_lib.get_rb_state(self._motion_ids,
                                                                (self.progress_buf + i + 1) * self.dt + self._motions_start_time
                                                                )
                rb_poses_gt_acc.append(rb_pos_gt[:, self._rigid_body_track_indices, :])
                rb_rots_gt_acc.append(rb_rot_gt[:, self._rigid_body_track_indices, :])
            rb_poses_gt_acc = torch.cat(rb_poses_gt_acc, dim=1)
            rb_rots_gt_acc = torch.cat(rb_rots_gt_acc, dim=1)

        body_pos = self._rigid_body_pos.clone()
        body_rot = self._rigid_body_rot.clone()
        body_vel = self._rigid_body_vel.clone()
        body_ang_vel = self._rigid_body_ang_vel.clone()
        dof_pos = self._dof_pos.clone()
        dof_vel = self._dof_vel.clone()
        reset_val = torch.zeros(len(body_pos), dtype=body_pos.dtype, device=body_pos.device)
        # the first time that is reset done the state tensors are not updated yet therefore we use the value of reference
        # instead
        if len(self.reset_envs_dict["env_ids"]) > 0:
            reset_val[self.reset_envs_dict["env_ids"]] = 1.0
            body_pos[self.reset_envs_dict["env_ids"]] = self.reset_envs_dict["body_pos"]
            body_rot[self.reset_envs_dict["env_ids"]] = self.reset_envs_dict["body_rot"]
            body_vel[self.reset_envs_dict["env_ids"]] = self.reset_envs_dict["body_vel"]
            body_ang_vel[self.reset_envs_dict["env_ids"]] = self.reset_envs_dict["body_ang_vel"]
            dof_pos[self.reset_envs_dict["env_ids"]] = self.reset_envs_dict["dof_pos"]
            dof_vel[self.reset_envs_dict["env_ids"]] = self.reset_envs_dict["dof_vel"]
            self.reset_envs_dict = {
                "env_ids": [],
                "body_pos": [],
                "body_rot": [],
                "body_vel": [],
                "body_ang_vel": [],
                "dof_pos": [],
                "dof_vel": []

            }




        if (env_ids is None):
            feet_contact_forces = self.feet_contact_forces
            env_rb_poses_gt_acc = rb_poses_gt_acc
            env_rb_rots_gt_acc = rb_rots_gt_acc
            imit_heights = self.imit_motion_heights
            humanoid_heights = self.humanoid_heights
            
        else:
            reset_val = reset_val[env_ids]
            body_pos = body_pos[env_ids]
            body_rot = body_rot[env_ids]
            body_vel = body_vel[env_ids]
            body_ang_vel = body_ang_vel[env_ids]
            dof_pos = dof_pos[env_ids]
            dof_vel = dof_vel[env_ids]
            feet_contact_forces = self.feet_contact_forces[env_ids]
            env_rb_poses_gt_acc = rb_poses_gt_acc[env_ids]
            env_rb_rots_gt_acc = rb_rots_gt_acc[env_ids]
            imit_heights = self.imit_motion_heights[env_ids]
            humanoid_heights = self.humanoid_heights[env_ids]

        self.check_is_valid(body_pos)
        self.check_is_valid(body_rot)
        self.check_is_valid(body_vel)
        self.check_is_valid(body_ang_vel)
        self.check_is_valid(self._rigid_body_joints_indices)
        self.check_is_valid(dof_pos)
        self.check_is_valid(dof_vel)
        self.check_is_valid(feet_contact_forces)
        self.check_is_valid(env_rb_poses_gt_acc)
        self.check_is_valid(env_rb_rots_gt_acc)
        self.check_is_valid(imit_heights)
        self.check_is_valid(humanoid_heights)

        if self.pre_obs:
            obs = env_obs_util.get_obs_pre(_rigid_body_pos=body_pos, _rigid_body_rot=body_rot, _rigid_body_vel=body_vel,
                                       _rigid_body_ang_vel=body_ang_vel,
                                       _rigid_body_joints_indices=self._rigid_body_joints_indices,
                                       dof_pos=dof_pos, dof_vel=dof_vel, feet_contact_forces=feet_contact_forces,
                                       track_poses_acc=env_rb_poses_gt_acc, track_rots_acc=env_rb_rots_gt_acc,
                                       track_headset_index=self._track_headset_index, num_track_dev=self.num_track_dev,
                                       imit_motion_height=imit_heights, humanoid_height=humanoid_heights,
                                       include_global=self.include_global_obs)
        else:
            obs = env_obs_util.get_obs(_rigid_body_pos=body_pos, _rigid_body_rot=body_rot, _rigid_body_vel=body_vel,
                                       _rigid_body_ang_vel=body_ang_vel,
                                       _rigid_body_joints_indices=self._rigid_body_joints_indices,
                                       dof_pos=dof_pos, dof_vel=dof_vel, feet_contact_forces=feet_contact_forces,
                                       track_poses_acc=env_rb_poses_gt_acc, track_rots_acc=env_rb_rots_gt_acc,
                                       track_headset_index=self._track_headset_index, num_track_dev=self.num_track_dev,
                                       reset_val=reset_val,
                                       imit_motion_height=imit_heights, humanoid_height=humanoid_heights,
                                       include_global=self.include_global_obs)

        return obs

    def check_is_valid(self, ten):
        if torch.any(torch.isnan(ten)).item():
            raise Exception("the tensor contains a nan")
        if torch.any(torch.isinf(ten)).item():
            raise Exception("the tensor contains a inf")

    def pre_physics_step(self, actions):
        if self.deb_sync or self.deb2:
            self.actions = actions.to(self.device).clone()
            forces = torch.zeros_like(self.actions)
            force_tensor = gymtorch.unwrap_tensor(forces)
            self.gym.set_dof_actuation_force_tensor(self.sim, force_tensor)
        else:
            super().pre_physics_step(actions)

    def post_physics_step(self):
        if self.deb_sync:
            self._motion_sync()
        super().post_physics_step()
        self.prev_feet_contact_forces[:] = torch.clone(self.feet_contact_forces)
        if self.cfg["env"]["real_time"]:
            self.extras["body_pos"] = self._rigid_body_pos
            self.extras["body_rot"] = self._rigid_body_rot

    #this ethod is not used for training here just for debug
    def _motion_sync(self):
        motion_ids = self._motion_ids
        motion_times = (self.progress_buf + 1) * self.dt + self._motions_start_time

        root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel, key_pos \
            = self._motion_lib.get_motion_state(motion_ids, motion_times)

        rb_pos, rb_rot, rb_vel, d_pos, d_vel, _ = self._motion_lib.get_rb_state(
            self._motion_ids, (self.progress_buf+1) * self.dt + self._motions_start_time)

        # root_vel = torch.zeros_like(root_vel)
        # root_ang_vel = torch.zeros_like(root_ang_vel)
        # dof_vel = torch.zeros_like(dof_vel)

        env_ids = torch.arange(self.num_envs, dtype=torch.long, device=self.device)
        self._set_env_state(env_ids=env_ids,
                            root_pos=root_pos,
                            root_rot=root_rot,
                            dof_pos=dof_pos,
                            root_vel=root_vel,
                            root_ang_vel=root_ang_vel,
                            dof_vel=dof_vel)

        self._set_rb_state(env_ids, rb_pos=rb_pos, rb_rot=rb_rot, rb_vel=rb_vel)

        env_ids_int32 = self._humanoid_actor_ids[env_ids]
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self._root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self._dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

    def _set_rb_state(self, env_ids, rb_pos, rb_rot, rb_vel, rb_ang_vel=None):
        self._rigid_body_pos[env_ids, :, :] = rb_pos
        self._rigid_body_rot[env_ids, :, :] = rb_rot
        self._rigid_body_vel[env_ids, :, :] = rb_vel

        if rb_ang_vel is not None:
            self._rigid_body_ang_vel[env_ids, :, :] = rb_ang_vel


    def _reset_envs(self, env_ids):
        super()._reset_envs(env_ids)
        if (len(env_ids) > 0):
            #It is important to do this after the reset of actors and call or refresh actors; so that the values in
            # the sim tensors are already updated
            reset_envs_feet_contact_forces = torch.zeros_like(self.prev_feet_contact_forces)
            #Currently reset to 0. It depends from start pose. Therefore set to 0.
            #There might be better way to reset this
            self.prev_feet_contact_forces[env_ids] = reset_envs_feet_contact_forces[env_ids]
        return

    # Extra methods for visualizing

    def render(self, sync_frame_time=False):
        super().render(sync_frame_time)

        if self.viewer:
            self.gym.clear_lines(self.viewer)
            #self._visualize_bodies_transforms(self._rigid_body_track_indices)
            #self._visualize_bodies_transforms(self._rigid_body_joints_indices, sphere_color=(0, 0, 1))
            feet_pos = self._rigid_body_pos[:, self._contact_feet_ids, :]
            self._visualize_force(feet_pos, self.feet_contact_forces)
            self._visualize_real_time_input()

    def _visualize_bodies_transforms(self, body_ids, sphere_color=None):
        positions = self._rigid_body_pos[:, body_ids, :]
        rotations = self._rigid_body_rot[:, body_ids, :]
        self._visualize_pose(positions, rotations, sphere_color)

    def _visualize_real_time_input(self):
        if self.cfg["env"]["real_time"]:
            if self.imitState.is_ready():
                val = self.imitState.get()

                val = val[0]  # the first element of the list (that is the first pose)
                # print(f"From main_loop: {val}")
                positions = torch.stack([val[0]] * self.num_envs, dim=0)
                rotations = torch.stack([val[1]] * self.num_envs, dim=0)
                self._visualize_pose(positions, rotations, sphere_color=(1, 0, 0))

    def _visualize_pose(self, position, rotation, sphere_color=None):
        # axes and sphere for transform
        axes_geom = gymutil.AxesGeometry(0.1)
        # Create a wireframe sphere
        sphere_rot = gymapi.Quat.from_euler_zyx(0.5 * math.pi, 0, 0)
        sphere_pose = gymapi.Transform(r=sphere_rot)
        if sphere_color is None:
            sphere_color = (1, 1, 0)
        sphere_geom = gymutil.WireframeSphereGeometry(0.02, 12, 12, sphere_pose, color=sphere_color)

        for i in range(self.num_envs):
            if position.ndim == 3:
                for j in range(position.shape[1]):
                    pv = gymapi.Vec3(position[i, j, 0], position[i, j, 1], position[i, j, 2])
                    pq = gymapi.Quat(rotation[i, j, 0], rotation[i, j, 1], rotation[i, j, 2], rotation[i, j, 3])
                    pose = gymapi.Transform(pv, pq)
                    gymutil.draw_lines(axes_geom, self.gym, self.viewer, self.envs[i], pose)
                    gymutil.draw_lines(sphere_geom, self.gym, self.viewer, self.envs[i], pose)
            else:  # just 2D env and one single pos
                pv = gymapi.Vec3(position[i, 0], position[i, 1], position[i, 2])
                pq = gymapi.Quat(rotation[i, 0], rotation[i, 1], rotation[i, 2], rotation[i, 3])
                pose = gymapi.Transform(pv, pq)
                gymutil.draw_lines(axes_geom, self.gym, self.viewer, self.envs[i], pose)
                gymutil.draw_lines(sphere_geom, self.gym, self.viewer, self.envs[i], pose)

    def _visualize_force(self, force_beg_pos, force):
        #print(force)
        for i in range(self.num_envs):
            extra_vf = 25
            extra_vf_sqrt = 5
            just_z = True
            if extra_vf < 2:
                num_lines = force[i].shape[0]
                begins = force_beg_pos[i].cpu().numpy()
                f = force[i]
                if just_z:
                    f[:, 0] = 0.
                    f[:, 1] = 0.
                normalized_force = torch.nn.functional.normalize(f, p=2, dim=1)
                ends = begins + (normalized_force.cpu().numpy())
            else:
                num_lines = force[i].shape[0] * extra_vf
                begins = force_beg_pos[i].cpu().numpy()
                begins = begins.repeat(extra_vf, axis=0)

                #creates displacement
                max_dist = 0.01
                displ = np.linspace(-max_dist, max_dist, num=extra_vf_sqrt)
                xs, ys = np.meshgrid(displ, displ)
                xs = xs.ravel()
                ys = ys.ravel()
                zeros_z = np.zeros((extra_vf))
                displ_v = np.stack((xs, ys, zeros_z), axis=-1)
                displ_v = np.concatenate([displ_v]*force[i].shape[0], axis=0)
                begins = begins + displ_v

                f = force[i]
                if just_z:
                    f[:, 0] = 0.
                    f[:, 1] = 0.
                normalized_force = torch.nn.functional.normalize(f, p=2, dim=1)
                normalized_force = normalized_force.cpu().numpy()
                normalized_force = normalized_force.repeat(extra_vf, axis=0)
                ends = begins + normalized_force
            verts = np.zeros(shape=(num_lines, 6), dtype=np.float32)
            verts[:, :3] = begins  # assigning the begin
            verts[:, 3:] = ends  # assigning ends
            verts = verts.ravel()
            self.gym.add_lines(self.viewer, self.envs[i], num_lines, verts,
                               np.array([1.0, 0., 0.] * num_lines, dtype=np.float32))

class HumanoidImitationTrackTest(HumanoidImitationTrack):
    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        super().__init__(cfg=cfg,
                         sim_params=sim_params,
                         physics_engine=physics_engine,
                         device_type=device_type,
                         device_id=device_id,
                         headless=headless)

        self.cfg["env"]["motion_lib"] = self._motion_lib

    def post_physics_step(self):
        super().post_physics_step()
        self.extras["body_pos"] = self._rigid_body_pos
        self.extras["body_rot"] = self._rigid_body_rot
        self.extras["body_vel"] = self._rigid_body_vel
        self.extras["body_ang_vel"] = self._rigid_body_ang_vel
        if self.use_multiple_heights:
            #we use the asset height in order to estimate the error correctly
            rb_pos_gt, rb_rot_gt, rb_vel_gt, \
                dof_pos_gt, dof_vel_gt, _ = self._motion_lib.get_rb_state(self._motion_ids,
                                                                       self.progress_buf * self.dt + self._motions_start_time,
                                                                       self.humanoid_heights)
        else:
            rb_pos_gt, rb_rot_gt, rb_vel_gt, \
                dof_pos_gt, dof_vel_gt, _ = self._motion_lib.get_rb_state(self._motion_ids,
                                                                       self.progress_buf * self.dt + self._motions_start_time)




        self.extras["body_pos_gt"] = rb_pos_gt
        self.extras["body_rot_gt"] = rb_rot_gt
        self.extras["body_vel_gt"] = rb_vel_gt

    def reset_with_motion(self, motion_ids, env_ids=None):
        self._state_init = HumanoidMotionAndReset.StateInit.Start
        if not torch.is_tensor(motion_ids):
            motion_ids = to_torch(motion_ids, dtype=self._motion_ids.dtype, device=self.device)

        num_ids_needed = self.num_envs if env_ids is None else env_ids.shape[0]
        if len(motion_ids.shape) == 0:
            motion_ids = torch.stack([motion_ids]*num_ids_needed)
        elif motion_ids.shape[0] != num_ids_needed:
            motion_ids = torch.cat([motion_ids]*num_ids_needed, dim=0)


        self.hard_reset_motion_ids = motion_ids

        if (env_ids is None):
            env_ids = to_torch(np.arange(self.num_envs), device=self.device, dtype=torch.long)
        self._reset_envs(env_ids)


@torch.jit.script
def compute_ref_motion_reset(reset_buf, motion_lengths, progress_buf, dt, motions_start_time):
    # type: (Tensor, Tensor, Tensor, float, Tensor) -> Tuple[Tensor, Tensor]
    terminated = torch.zeros_like(reset_buf)
    motion_times = progress_buf * dt + motions_start_time
    reset = torch.where(motion_times > motion_lengths, torch.ones_like(reset_buf), terminated)
    return reset, terminated