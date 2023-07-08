from learning.common_player import CommonPlayerWithWriter

import torch

import time
from isaacgym.torch_utils import quat_mul, quat_conjugate, quat_unit
from utils.torch_utils import quat_to_angle_axis

class CommonTester(CommonPlayerWithWriter):
    def __init__(self, config):
        super().__init__(config)

        # lists for storing the values
        self.accumulated_vels = []
        self.accumulated_pos_error = []
        self.accumulated_rot_error = []
        self.accumulated_loc_error = []
        self.accumulated_positions = []
        self.accumulated_positions_gt = []
        # dict for storing the values that must be averaged over all the motion
        self.measures_motions_vals = {}

    def run(self):
        render = self.render_env
        n_game_life = self.n_game_life
        is_determenistic = self.is_determenistic
        sum_rewards = 0
        sum_steps = 0
        sum_weighted_rewards = 0
        games_played = 0
        has_masks = False
        has_masks_func = getattr(self.env, "has_action_mask", None) is not None

        op_agent = getattr(self.env, "create_agent", None)
        if op_agent:
            agent_inited = True

        if has_masks_func:
            has_masks = self.env.has_action_mask()


        env_motion_lib = self.env_config["env"]["motion_lib"]
        num_motions = env_motion_lib.num_motions()
        for m_i in range(num_motions):
            obs_dict = self.env_reset_with_motion(motion_ids=m_i)
            m_i_name = env_motion_lib.get_motion_name(motion_id=m_i)
            batch_size = 1
            batch_size = self.get_batch_size(obs_dict['obs'], batch_size)
            cr = torch.zeros(batch_size, dtype=torch.float32, device=self.device)
            steps = torch.zeros(batch_size, dtype=torch.float32, device=self.device)

            motion_length = env_motion_lib.get_motion_length(m_i)
            steps_required = motion_length // self.env.task.dt
            steps_required += 1 # to compensate rounding errors
            steps_required = int(steps_required.item())
            for n in range(steps_required):
                # we do not check if done or not. We are evaluating motion imitation and run the loop
                # as many steps as the motion requires
                obs_dict = self.env_reset([])

                if has_masks:
                    masks = self.env.get_action_mask()
                    action = self.get_masked_action(obs_dict, masks, is_determenistic)
                else:
                    action = self.get_action(obs_dict, is_determenistic)
                obs_dict, r, done, info = self.env_step(self.env, action)
                #print(f"step: {n}")

                cr += r
                steps += 1
                info["reward"] = r.mean().item()


                self._log_env_step(m_i_name, info, n)
                self._post_step(info)

                if render:
                    self.env.render(mode='human')
                    time.sleep(self.render_sleep)

            games_played += 1

            # we do not check if done or not. We are evaluating motion imitation and run the loop
            # as many steps as the motion requires
            cur_reward = cr.mean().item()
            cur_steps = steps.mean().item()
            weighted_cur_reward = cur_reward / cur_steps
            self.game_cumulated_reward = cur_reward
            self.game_cumulated_weighted_reward = weighted_cur_reward
            sum_rewards += cur_reward
            sum_steps += cur_steps
            sum_weighted_rewards += weighted_cur_reward
            print('reward:', cur_reward, 'steps:', cur_steps, 'weighted reward:', weighted_cur_reward)

            # order is important due to operations of post
            self._evaluate_game(m_i_name)
            self._post_game()


        print(sum_rewards)
        print(sum_weighted_rewards)
        print('av reward:', sum_rewards / games_played * n_game_life, 'av steps:',
              sum_steps / games_played * n_game_life, 'av weighted reward:',
              sum_weighted_rewards / games_played * n_game_life)

        av_reward = sum_rewards / games_played
        av_weighted_reward = sum_weighted_rewards / games_played
        self.writer.add_scalar("am_cumulated_reward", av_reward)
        self.writer.add_scalar("am_cumulated_weighted_reward", av_weighted_reward)
        self._post_test()

    def env_reset_with_motion(self, motion_ids, env_ids=None):
        obs = self.env.reset_with_motion(motion_ids, env_ids)
        return self.obs_to_torch(obs)

    def _post_step(self, info):
        self.accumulated_vels.append(info["body_vel"])
        pos_error = info["body_pos_gt"] - info["body_pos"]
        pos_error = pos_error.norm(dim=2)
        self.accumulated_pos_error.append(pos_error)
        #loc error
        loc_error = info["body_pos_gt"][:, 0, :2] - info["body_pos"][:, 0, :2]
        loc_error = loc_error.norm(dim=1)
        self.accumulated_loc_error.append(loc_error)
        #positions
        self.accumulated_positions.append(info["body_pos"])
        self.accumulated_positions_gt.append(info["body_pos_gt"])

        # estimates angle error
        q1 = info["body_rot"]
        q2 = info["body_rot_gt"]
        original_shape = q1.shape
        #first we reshape since quaternion function expect shpae of (N, 4)
        q1 = q1.reshape(q1.shape[0] * q1.shape[1], q1.shape[2])
        q2 = q2.reshape(q2.shape[0] * q2.shape[1], q2.shape[2])
        # quaternion difference
        q_delta = quat_mul(q2, quat_conjugate(q1))
        angles_error, _ = quat_to_angle_axis(q_delta)
        angles_error = angles_error.reshape(original_shape[0], original_shape[1])
        self.accumulated_rot_error.append(angles_error)

    def _post_game(self):
        self.accumulated_vels = []
        self.accumulated_pos_error = []
        self.accumulated_rot_error = []
        self.accumulated_loc_error = []
        self.accumulated_positions = []
        self.accumulated_positions_gt = []

    def _log_env_step(self, motion_name, info, step):
        #log positions
        joints_indices = self.env_config["env"]["asset"]["jointsIndices"]
        joints_names = self.env_config["env"]["asset"]["jointsNames"]
        num_envs = info["body_pos"].shape[0]

    def _evaluate_game(self, motion_name):
        self.accumulated_vels = torch.stack(self.accumulated_vels)


        self.accumulated_pos_error = torch.stack(self.accumulated_pos_error)
        self.accumulated_rot_error = torch.stack(self.accumulated_rot_error)

        self.accumulated_loc_error = torch.stack(self.accumulated_loc_error)
        self.accumulated_positions = torch.stack(self.accumulated_positions)
        self.accumulated_positions_gt = torch.stack(self.accumulated_positions_gt)


        # mean over all parallel envs
        self.accumulated_pos_error = torch.mean(self.accumulated_pos_error, dim=1)
        self.accumulated_rot_error = torch.mean(self.accumulated_rot_error, dim=1)
        self.accumulated_loc_error = torch.mean(self.accumulated_loc_error, dim=1)

        #putting the measure of the position error in cm
        self.accumulated_pos_error *= 100.0

        joints_indices = self.env_config["env"]["asset"]["jointsIndices"]
        joints_names = self.env_config["env"]["asset"]["jointsNames"]

        track_indices = self.env_config["env"]["asset"]["trackIndices"]
        track_names = self.env_config["env"]["asset"]["trackBodies"]

        def log_jerk():
            # Estimate jitter; measured by jerk. Jerk is third time derivative of positions, second derivative of vel

            jerk = self.accumulated_positions[3:] - 3*self.accumulated_positions[2:-1] +\
                   3*self.accumulated_positions[1:-2] - self.accumulated_positions[:-3]
            jerk_gt = self.accumulated_positions_gt[3:] - 3 * self.accumulated_positions_gt[2:-1] + \
                   3 * self.accumulated_positions_gt[1:-2] - self.accumulated_positions_gt[:-3]
            dt = self.env.task.dt
            jerk = jerk / (dt**3)

            jerk_gt = jerk_gt / (dt**3)

            jerk = jerk.norm(dim=3)  # calculates the norm of the jerk vector
            jerk = torch.mean(jerk, dim=1) # mean over all parallel envs

            jerk_gt = jerk_gt.norm(dim=3)
            jerk_gt = torch.mean(jerk_gt, dim=1)

            #calculate average over all joints and steps
            jerk = jerk[:, joints_indices]  # selects the joints
            jerk = torch.mean(jerk, dim=1)
            jerk = torch.mean(jerk, dim=0)

            jerk_gt = jerk_gt[:, joints_indices]  # selects the joints
            jerk_gt = torch.mean(jerk_gt, dim=1)
            jerk_gt = torch.mean(jerk_gt, dim=0)

            # todo ?? putting the measure of jerk in (km/s^3)
            #jerk /= 1000.0
            average_name = "as_aj_jitter"
            average_name_gt = "as_aj_jitter_gt"
            # self.writer.add_scalar(
            #     f'{motion_name}-{average_name}', jerk, 0
            # )
            if average_name not in self.measures_motions_vals.keys():
                self.measures_motions_vals[average_name] = []
            if average_name_gt not in self.measures_motions_vals.keys():
                self.measures_motions_vals[average_name_gt] = []
            self.measures_motions_vals[average_name].append(jerk)
            self.measures_motions_vals[average_name_gt].append(jerk_gt)

        def log_error():
            accumulated_pos_error = self.accumulated_pos_error
            accumulated_rot_error = self.accumulated_rot_error

            # for step in range(accumulated_pos_error.shape[0]):
            #     scalar_info_tuple_list = [
            #         (f'{motion_name}-pos_error', accumulated_pos_error[step]),
            #         (f'{motion_name}-rot_error', accumulated_rot_error[step])
            #     ]
            #     self.__joints_or_track_log_loop(els_indices=joints_indices, els_names=joints_names, step=step,
            #                                     scalar_info_tuple_list=scalar_info_tuple_list)

            mean_pos_error_per_joint = torch.mean(accumulated_pos_error, dim=0)
            mean_rot_error_per_joint = torch.mean(accumulated_rot_error, dim=0)
            # mean_per_joint_info_tuple_list = [
            #     (f'{motion_name}-as_pos_error/', mean_pos_error_per_joint),
            #     (f'{motion_name}-as_rot_error/', mean_rot_error_per_joint)
            # ]
            # self.__joints_or_track_log_loop(els_indices=joints_indices, els_names=joints_names, step=0,
            #                                 scalar_info_tuple_list=mean_per_joint_info_tuple_list)

            mean_pos_error_per_joint = mean_pos_error_per_joint[joints_indices]  # selects the joints
            mean_rot_error_per_joint = mean_rot_error_per_joint[joints_indices]
            mean_pos_error = torch.mean(mean_pos_error_per_joint)
            mean_rot_error = torch.mean(mean_rot_error_per_joint)
            average_name_pos = "as_aj_pos_error"
            average_name_rot = "as_aj_rot_error"
            # self.writer.add_scalar(
            #     f'{motion_name}-{average_name_pos}',
            #     mean_pos_error, 0
            # )
            # self.writer.add_scalar(
            #     f'{motion_name}-{average_name_rot}',
            #     mean_rot_error, 0
            # )
            if average_name_pos not in self.measures_motions_vals.keys():
                self.measures_motions_vals[average_name_pos] = []
            if average_name_rot not in self.measures_motions_vals.keys():
                self.measures_motions_vals[average_name_rot] = []
            self.measures_motions_vals[average_name_pos].append(mean_pos_error)
            self.measures_motions_vals[average_name_rot].append(mean_rot_error)

        def log_error_trackers():
            accumulated_pos_error = self.accumulated_pos_error  # selects the tracking devices
            accumulated_rot_error = self.accumulated_rot_error  # selects the tracking devices
            # for step in range(accumulated_pos_error.shape[0]):
            #     for i in range(len(track_indices)):
            #         j_id = track_indices[i]
            #         j_name = track_names[i]
            #         self.writer.add_scalar(
            #             f'{motion_name}-pos_error_track/{j_name}',
            #             accumulated_pos_error[step, j_id], step
            #         )
            #         self.writer.add_scalar(
            #             f'{motion_name}-rot_error_track/{j_name}',
            #             accumulated_rot_error[step, j_id], step
            #         )

            mean_pos_error_per_joint = torch.mean(accumulated_pos_error, dim=0)
            mean_rot_error_per_joint = torch.mean(accumulated_rot_error, dim=0)
            # for i in range(len(track_indices)):
            #     j_id = track_indices[i]
            #     j_name = track_names[i]
            #     self.writer.add_scalar(
            #         f'{motion_name}-as_pos_error_track/{j_name}',
            #         mean_pos_error_per_joint[j_id], 0
            #     )
            #     self.writer.add_scalar(
            #         f'{motion_name}-as_rot_error_track/{j_name}',
            #         mean_rot_error_per_joint[j_id], 0
            #     )
            mean_pos_error_per_joint = mean_pos_error_per_joint[track_indices]  # selects the tracking devices
            mean_rot_error_per_joint = mean_rot_error_per_joint[track_indices]
            mean_pos_error = torch.mean(mean_pos_error_per_joint)
            mean_rot_error = torch.mean(mean_rot_error_per_joint)

            average_name_pos = "as_aj_pos_error_track"
            average_name_rot = "as_aj_rot_error_track"
            # self.writer.add_scalar(
            #     f'{motion_name}-{average_name_pos}',
            #     mean_pos_error, 0
            # )
            # self.writer.add_scalar(
            #     f'{motion_name}-{average_name_rot}',
            #     mean_rot_error, 0
            # )
            if average_name_pos not in self.measures_motions_vals.keys():
                self.measures_motions_vals[average_name_pos] = []
            if average_name_rot not in self.measures_motions_vals.keys():
                self.measures_motions_vals[average_name_rot] = []
            self.measures_motions_vals[average_name_pos].append(mean_pos_error)
            self.measures_motions_vals[average_name_rot].append(mean_rot_error)

        def log_sip():
            #the extrmities used according to QuestSim
            extremities_names = ["right_upper_arm", "left_upper_arm", "right_thigh", "left_thigh"]
            extremities_joints_indices = []
            for name in extremities_names:
                i = joints_names.index(name)
                extremities_joints_indices.append(joints_indices[i])

            #select the measures of the extremities
            accumulated_rot_error = self.accumulated_rot_error[:, extremities_joints_indices]
            sip = torch.mean(accumulated_rot_error, dim=1)

            # for step in range(sip.shape[0]):
            #     self.writer.add_scalar(
            #         f'{motion_name}-sip', sip[step], step
            #     )
            mean_step_sip = torch.mean(sip)
            average_name = "as_sip"
            # self.writer.add_scalar(
            #     f'{motion_name}-{average_name}', mean_step_sip, 0
            # )
            if average_name not in self.measures_motions_vals.keys():
                self.measures_motions_vals[average_name] = []
            self.measures_motions_vals[average_name].append(mean_step_sip)

        def log_reward():
            # log rewards
            # self.writer.add_scalar(f'{motion_name}-cumulated_reward', self.game_cumulated_reward)
            # self.writer.add_scalar(f'{motion_name}-cumulated_weighted_reward', self.game_cumulated_weighted_reward)
            pass
            # here we do not add to measures_motions_vals since we already estimate the average over motions
            # in run()

        def log_loc_error():
            accumulated_loc_error = self.accumulated_loc_error
            mean_loc_error = torch.mean(accumulated_loc_error, dim=0)
            average_name_loc = "as_loc_error"
            #putting in cm
            mean_loc_error *= 100.0
            if average_name_loc not in self.measures_motions_vals.keys():
                self.measures_motions_vals[average_name_loc] = []
            self.measures_motions_vals[average_name_loc].append(mean_loc_error)

        log_jerk()
        log_error()
        log_error_trackers()
        log_sip()
        log_reward()
        log_loc_error()

    def _post_test(self):
        for measure_name, measure_list in self.measures_motions_vals.items():
            l = len(measure_list)
            average = sum(measure_list)/l
            self.writer.add_scalar(f"am_{measure_name}", average, 0)





    # def __joints_or_track_log_loop(self, els_indices, els_names, step, scalar_info_tuple_list):
    #     for i in range(len(els_indices)):
    #         el_id = els_indices[i]
    #         el_name = els_names[i]
    #         for (scalar_name_prefix, scalar_info) in scalar_info_tuple_list:
    #             self.writer.add_scalar(
    #                 f'{scalar_name_prefix}/{el_name}',
    #                 scalar_info[el_id], step
    #             )






