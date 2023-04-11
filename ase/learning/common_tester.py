from learning.common_player import CommonPlayer

import torch
from tensorboardX import SummaryWriter

import time
import os
from datetime import datetime
import re
from utils.motion_lib import MotionLib
from isaacgym.torch_utils import quat_mul, quat_conjugate, quat_unit
from utils.torch_utils import quat_to_angle_axis


class CommonTester(CommonPlayer):
    def __init__(self, config):
        super().__init__(config)
        # creates writer for test results
        # tries to create filename based on the checkpoint
        checkpoint_path_els = self.config["checkpoint"].split(os.sep)
        # checks that a checkpoint is given and start with the same path as the path where we want to write results
        if len(checkpoint_path_els) > 0 and self.config["checkpoint"].startswith(self.config["train_dir"]):
            exp_type_name = self.config["name"]
            # check if it is potentially the same that we are testing
            pattern = re.compile(f"{exp_type_name}_[0-9][0-9]-[0-9][0-9]-[0-9][0-9]-[0-9][0-9]")
            if len(checkpoint_path_els) > 2 and pattern.match(checkpoint_path_els[1]):
                self.test_results_dir = self.config["train_dir"] + os.sep + checkpoint_path_els[
                    1] + os.sep + "test_results" + os.sep
            else:
                self.test_results_dir = self.config["train_dir"] + os.sep + "test_results" + os.sep
        else:
            self.test_results_dir = self.config["train_dir"] + os.sep + "test_results" + os.sep

        self.test_results_dir = self.test_results_dir + datetime.now().strftime("_%d-%H-%M-%S")

        os.makedirs(self.test_results_dir, exist_ok=True)
        self.writer = SummaryWriter(self.test_results_dir)

    def run(self):
        render = self.render_env
        n_game_life = self.n_game_life
        is_determenistic = self.is_determenistic
        sum_rewards = 0
        sum_steps = 0
        sum_game_res = 0
        games_played = 0
        has_masks = False
        has_masks_func = getattr(self.env, "has_action_mask", None) is not None

        op_agent = getattr(self.env, "create_agent", None)
        if op_agent:
            agent_inited = True

        if has_masks_func:
            has_masks = self.env.has_action_mask()

        need_init_rnn = self.is_rnn

        env_motion_lib :MotionLib = self.env_config["env"]["motion_lib"]
        num_motions = env_motion_lib.num_motions()
        for m_i in range(num_motions):
            obs_dict = self.env_reset_with_motion(motion_ids=m_i)
            m_i_name = env_motion_lib.get_motion_name(motion_id=m_i)
            batch_size = 1
            batch_size = self.get_batch_size(obs_dict['obs'], batch_size)

            if need_init_rnn:
                self.init_rnn()
                need_init_rnn = False

            cr = torch.zeros(batch_size, dtype=torch.float32, device=self.device)
            steps = torch.zeros(batch_size, dtype=torch.float32, device=self.device)

            print_game_res = False

            done_indices = []

            motion_length = env_motion_lib.get_motion_length(m_i)
            steps_required = motion_length // self.env.task.dt
            steps_required += 1 # to compensate rounding
            steps_required = int(steps_required.item())
            for n in range(steps_required):
                obs_dict = self.env_reset(done_indices)

                if has_masks:
                    masks = self.env.get_action_mask()
                    action = self.get_masked_action(obs_dict, masks, is_determenistic)
                else:
                    action = self.get_action(obs_dict, is_determenistic)
                obs_dict, r, done, info = self.env_step(self.env, action)
                cr += r
                steps += 1

                self._log_env_step(m_i_name, info, n)
                self._post_step(info)

                if render:
                    self.env.render(mode='human')
                    time.sleep(self.render_sleep)

                all_done_indices = done.nonzero(as_tuple=False)
                done_indices = all_done_indices[::self.num_agents]
                done_count = len(done_indices)
                #games_played += done_count

                if done_count > 0:
                    if self.is_rnn:
                        for s in self.states:
                            s[:, all_done_indices, :] = s[:, all_done_indices, :] * 0.0

                    cur_rewards = cr[done_indices].sum().item()
                    cur_steps = steps[done_indices].sum().item()

                    cr = cr * (1.0 - done.float())
                    steps = steps * (1.0 - done.float())
                    sum_rewards += cur_rewards
                    sum_steps += cur_steps

                    game_res = 0.0
                    if isinstance(info, dict):
                        if 'battle_won' in info:
                            print_game_res = True
                            game_res = info.get('battle_won', 0.5)
                        if 'scores' in info:
                            print_game_res = True
                            game_res = info.get('scores', 0.5)
                    if self.print_stats:
                        if print_game_res:
                            print('reward:', cur_rewards / done_count, 'steps:', cur_steps / done_count, 'w:', game_res)
                        else:
                            print('reward:', cur_rewards / done_count, 'steps:', cur_steps / done_count)

                    sum_game_res += game_res
                    if batch_size // self.num_agents == 1:
                        break

                done_indices = done_indices[:, 0]

            # order is important due to operations of post
            self._log_game(m_i_name)
            self._post_game()
            games_played+=1# todo 1 or the number of envs?

        print(sum_rewards)
        if print_game_res:
            print('av reward:', sum_rewards / games_played * n_game_life, 'av steps:',
                  sum_steps / games_played * n_game_life, 'winrate:', sum_game_res / games_played * n_game_life)
        else:
            print('av reward:', sum_rewards / games_played * n_game_life, 'av steps:',
                  sum_steps / games_played * n_game_life)

    def env_reset_with_motion(self, motion_ids, env_ids=None):
        obs = self.env.reset_with_motion(motion_ids, env_ids)
        return self.obs_to_torch(obs)

    def _post_step(self, info):
        if not hasattr(self, "accumulated_vels"):
            self.accumulated_vels = []
        self.accumulated_vels.append(info["body_vel"])
        if not hasattr(self, "accumulated_pos_error"):
            self.accumulated_pos_error = []
        if not hasattr(self, "accumulated_rot_error"):
            self.accumulated_rot_error = []
        pos_error = info["body_pos_gt"] - info["body_pos"]
        pos_error = pos_error.norm(dim=2)
        self.accumulated_pos_error.append(pos_error)
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
        angles_error = angles_error.reshape(original_shape[0], original_shape[1], 1)
        self.accumulated_rot_error.append(angles_error)

    def _post_game(self):
        self.accumulated_vels = []
        self.accumulated_pos_error = []
        self.accumulated_rot_error = []

    def _log_env_step(self, motion_name, info, step):
        joints_indices = self.env_config["env"]["asset"]["jointsIndices"]
        joints_names = self.env_config["env"]["asset"]["jointsNames"]
        num_envs = info["body_pos"].shape[0]
        for n in range(num_envs):
            for i in range(len(joints_indices)):
                j_id = joints_indices[i]
                j_name = joints_names[i]
                self.writer.add_scalar(
                    f'position_x/{motion_name}/env_{n}/{j_name}',
                    info['body_pos'][n, j_id, 0], step)
                self.writer.add_scalar(
                    f'position_y/{motion_name}/env_{n}/{j_name}',
                    info['body_pos'][n, j_id, 1], step)
                self.writer.add_scalar(
                    f'position_z/{motion_name}/env_{n}/{j_name}',
                    info['body_pos'][n, j_id, 2], step)

    def _log_game(self, motion_name):
        self.accumulated_vels = torch.stack(self.accumulated_vels)
        self.accumulated_pos_error = torch.stack(self.accumulated_pos_error)
        self.accumulated_rot_error = torch.stack(self.accumulated_rot_error)

        def log_jerk():
            joints_indices = self.env_config["env"]["asset"]["jointsIndices"]
            joints_names = self.env_config["env"]["asset"]["jointsNames"]
            # Estimate jitter; measured by jerk. Jerk is third time derivative of positions, second derivative of vvel
            jerk = self.accumulated_vels[2:] - 2 * self.accumulated_vels[1:-1] + self.accumulated_vels[:-2]
            jerk = jerk / (self.env.task.dt * self.env.task.dt)
            jerk = jerk.norm(dim=3)#calulates the norm of the jerk vector

            for step in range(jerk.shape[0]):
                for n in range(jerk.shape[1]):
                    for i in range(len(joints_indices)):
                        j_id = joints_indices[i]
                        j_name = joints_names[i]
                        self.writer.add_scalar(
                            f'{motion_name}/env_{n}/{j_name}/jitter', jerk[step, n, j_id], step
                        )
            #calculate average over all joints
            jerk = jerk[:, :, joints_indices]  # selects the joints
            jerk = torch.mean(jerk, dim=2) # todo estimate average or sum?
            for step in range(jerk.shape[0]):
                for n in range(jerk.shape[1]):
                    self.writer.add_scalar(
                        f'{motion_name}/env_{n}/aj_jitter', jerk[step, n], step
                    )
            # calculate average over all steps
            jerk = torch.mean(jerk, dim=0)  # todo estimate average or sum?
            for n in range(jerk.shape[0]):
                self.writer.add_scalar(
                    f'{motion_name}/env_{n}/as_aj_jitter', jerk[n], 0
                )

        def log_error():
            joints_indices = self.env_config["env"]["asset"]["jointsIndices"]
            joints_names = self.env_config["env"]["asset"]["jointsNames"]
            accumulated_pos_error = self.accumulated_pos_error
            accumulated_rot_error = self.accumulated_rot_error

            for step in range(accumulated_pos_error.shape[0]):
                for n in range(accumulated_pos_error.shape[1]):
                    for i in range(len(joints_indices)):
                        j_id = joints_indices[i]
                        j_name = joints_names[i]
                        self.writer.add_scalar(
                            f'{motion_name}/env_{n}/{j_name}/pos_error',
                            accumulated_pos_error[step, n, j_id], step
                        )
                        self.writer.add_scalar(
                            f'{motion_name}/env_{n}/{j_name}/rot_error',
                            accumulated_rot_error[step, n, j_id], step
                        )


            mean_pos_error_per_joint = torch.mean(accumulated_pos_error, dim=0)
            mean_rot_error_per_joint = torch.mean(accumulated_rot_error, dim=0)
            for n in range(mean_pos_error_per_joint.shape[0]):
                for i in range(len(joints_indices)):
                    j_id = joints_indices[i]
                    j_name = joints_names[i]
                    self.writer.add_scalar(
                        f'{motion_name}/env_{n}/{j_name}/as_pos_error',
                        mean_pos_error_per_joint[n, j_id], 0
                    )
                    self.writer.add_scalar(
                        f'{motion_name}/env_{n}/{j_name}/as_rot_error',
                        mean_rot_error_per_joint[n, j_id], 0
                    )

            mean_pos_error_per_joint = mean_pos_error_per_joint[:, joints_indices]  # selects the joints
            mean_rot_error_per_joint = mean_rot_error_per_joint[:, joints_indices]
            mean_pos_error = torch.mean(mean_pos_error_per_joint, dim=1)
            mean_rot_error = torch.mean(mean_rot_error_per_joint, dim=1)
            for n in range(mean_pos_error.shape[0]):
                self.writer.add_scalar(
                    f'{motion_name}/env_{n}/as_aj_pos_error',
                    mean_pos_error[n], 0
                )
                self.writer.add_scalar(
                    f'{motion_name}/env_{n}/as_aj_rot_error',
                    mean_rot_error[n], 0
                )

        def log_error_trackers():
            track_indices = self.env_config["env"]["asset"]["trackIndices"]
            track_names = self.env_config["env"]["asset"]["trackBodies"]
            accumulated_pos_error = self.accumulated_pos_error  # selects the tracking devices
            accumulated_rot_error = self.accumulated_rot_error  # selects the tracking devices

            for step in range(accumulated_pos_error.shape[0]):
                for n in range(accumulated_pos_error.shape[1]):
                    for i in range(len(track_indices)):
                        j_id = track_indices[i]
                        j_name = track_names[i]
                        self.writer.add_scalar(
                            f'{motion_name}/env_{n}/{j_name}/pos_error_track',
                            accumulated_pos_error[step, n, j_id], step
                        )
                        self.writer.add_scalar(
                            f'{motion_name}/env_{n}/{j_name}/rot_error_track',
                            accumulated_rot_error[step, n, j_id], step
                        )

            mean_pos_error_per_joint = torch.mean(accumulated_pos_error, dim=0)
            mean_rot_error_per_joint = torch.mean(accumulated_rot_error, dim=0)
            for n in range(mean_pos_error_per_joint.shape[0]):
                for i in range(len(track_indices)):
                    j_id = track_indices[i]
                    j_name = track_names[i]
                    self.writer.add_scalar(
                        f'{motion_name}/env_{n}/{j_name}/as_pos_error_track',
                        mean_pos_error_per_joint[n, j_id], 0
                    )
                    self.writer.add_scalar(
                        f'{motion_name}/env_{n}/{j_name}/as_rot_error_track',
                        mean_rot_error_per_joint[n, j_id], 0
                    )
            mean_pos_error_per_joint = mean_pos_error_per_joint[:, track_indices]  # selects the tracking devices
            mean_rot_error_per_joint = mean_rot_error_per_joint[:, track_indices]
            mean_pos_error = torch.mean(mean_pos_error_per_joint, dim=1)
            mean_rot_error = torch.mean(mean_rot_error_per_joint, dim=1)
            for n in range(mean_pos_error.shape[0]):
                self.writer.add_scalar(
                    f'{motion_name}/env_{n}/as_aj_pos_error_track',
                    mean_pos_error[n], 0
                )
                self.writer.add_scalar(
                    f'{motion_name}/env_{n}/as_aj_rot_error_track',
                    mean_rot_error[n], 0
                )

        def log_sip():
            #the extrmities used according to QuestSim
            extremities_names = ["right_upper_arm", "left_upper_arm", "right_thigh", "left_thigh"]
            joints_indices = self.env_config["env"]["asset"]["jointsIndices"]
            joints_names = self.env_config["env"]["asset"]["jointsNames"]
            extremities_joints_indices = []
            for name in extremities_names:
                i = joints_names.index(name)
                extremities_joints_indices.append(joints_indices[i])

            #select the measures of the extremities
            accumulated_rot_error = self.accumulated_rot_error[:, :, extremities_joints_indices]
            sip = torch.mean(accumulated_rot_error, dim=2)

            for step in range(sip.shape[0]):
                for n in range(sip.shape[1]):
                    self.writer.add_scalar(
                        f'{motion_name}/env_{n}/sip', sip[step, n], step
                    )
            mean_step_sip = torch.mean(sip, dim=0)
            for n in range(mean_step_sip.shape[0]):
                self.writer.add_scalar(
                    f'{motion_name}/env_{n}/as_sip', mean_step_sip[n], 0
                )


        log_jerk()
        log_error()
        log_error_trackers()
        log_sip()




