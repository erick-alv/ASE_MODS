from learning.common_player import CommonPlayerWithWriter
import torch

from isaacgym.torch_utils import quat_mul, quat_conjugate, quat_unit
from utils.torch_utils import quat_to_angle_axis

import time


class CommonRealTimePlayer(CommonPlayerWithWriter):
    def __init__(self, config):
        super().__init__(config)

    def run(self):
        n_games = self.games_num
        render = self.render_env
        n_game_life = self.n_game_life
        is_determenistic = self.is_determenistic
        sum_rewards = 0
        sum_steps = 0
        sum_game_res = 0
        n_games = n_games * n_game_life
        games_played = 0
        has_masks = False
        has_masks_func = getattr(self.env, "has_action_mask", None) is not None

        op_agent = getattr(self.env, "create_agent", None)
        if op_agent:
            agent_inited = True

        if has_masks_func:
            has_masks = self.env.has_action_mask()

        need_init_rnn = self.is_rnn
        for _ in range(n_games):
            if games_played >= n_games:
                break

            obs_dict = self.env_reset()
            batch_size = 1
            batch_size = self.get_batch_size(obs_dict['obs'], batch_size)

            if need_init_rnn:
                self.init_rnn()
                need_init_rnn = False

            cr = torch.zeros(batch_size, dtype=torch.float32, device=self.device)
            steps = torch.zeros(batch_size, dtype=torch.float32, device=self.device)

            print_game_res = False

            done_indices = []

            start_time = time.time()
            for n in range(self.max_steps):  # todo see how to readjust this according to real time
                # TODO make here a handle user input
                # currently used just to call reset when user requires
                lastUserInput = self.env_config["env"]["imitState"].getLast()
                if lastUserInput is not None:
                    buttonPressB = lastUserInput[2][1]
                    if buttonPressB == 1.0:
                        # call first reset on imitState
                        self.env_config["env"]["imitState"].reset()
                        obs_dict = self.env_reset()
                    else:
                        obs_dict = self.env_reset(done_indices)
                else:
                    obs_dict = self.env_reset(done_indices)

                if has_masks:
                    masks = self.env.get_action_mask()
                    action = self.get_masked_action(obs_dict, masks, is_determenistic)
                else:
                    action = self.get_action(obs_dict, is_determenistic)
                obs_dict, r, done, info = self.env_step(self.env, action)
                cr += r
                steps += 1

                info["step"] = n
                self._post_step(info)

                if render:
                    self.env.render(mode='human')
                    time.sleep(self.render_sleep)

                all_done_indices = done.nonzero(as_tuple=False)
                done_indices = all_done_indices[::self.num_agents]
                done_count = len(done_indices)
                games_played += done_count

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
                    if batch_size // self.num_agents == 1 or games_played >= n_games:
                        break

                done_indices = done_indices[:, 0]

                end_time = time.time()
                # print(f"in main {end_time - start_time}")
                start_time = time.time()

        print(sum_rewards)
        if print_game_res:
            print('av reward:', sum_rewards / games_played * n_game_life, 'av steps:',
                  sum_steps / games_played * n_game_life, 'winrate:', sum_game_res / games_played * n_game_life)
        else:
            print('av reward:', sum_rewards / games_played * n_game_life, 'av steps:',
                  sum_steps / games_played * n_game_life)

        return

    def _post_step(self, info):
        track_indices = self.env_config["env"]["asset"]["trackIndices"]
        track_pos = info["body_pos"][:, track_indices, :]
        track_rot = info["body_pos"][:, track_indices, :]
        track_pos_gt = info["track_pos_gt"]
        track_rot_gt = info["track_rot_gt"]
        # get just the first readings of the input signal
        num_track_devices = len(track_indices)
        track_pos_gt = track_pos_gt[:, :num_track_devices, :]
        track_rot_gt = track_rot_gt[:, :num_track_devices, :]
        # pos error
        pos_error = track_pos_gt - track_pos
        pos_error = pos_error.norm(dim=2)
        # rot error
        q1 = track_rot
        q2 = track_rot_gt
        original_shape = q1.shape
        # first we reshape since quaternion function expect shpae of (N, 4)
        q1 = q1.reshape(q1.shape[0] * q1.shape[1], q1.shape[2])
        q2 = q2.reshape(q2.shape[0] * q2.shape[1], q2.shape[2])
        # quaternion difference
        q_delta = quat_mul(q2, quat_conjugate(q1))
        angles_error, _ = quat_to_angle_axis(q_delta)
        angles_error = angles_error.reshape(original_shape[0], original_shape[1])

        # logging the errors
        # estimates mean over all parallel envs
        pos_error = torch.mean(pos_error, dim=0)
        angles_error = torch.mean(angles_error, dim=0)
        # putting error in cm
        pos_error += 100

        track_names = track_indices = self.env_config["env"]["asset"]["trackBodies"]
        for i in range(len(track_indices)):
            device_name = track_names[i]
            self.writer.add_scalar(
                f'real_time-pos_error_track/{device_name}', pos_error[i], info["step"]
            )
            self.writer.add_scalar(
                f'real_time-rot_error_track/{device_name}', angles_error[i], info["step"]
            )
        # logging mean over all devices
        pos_error = torch.mean(pos_error)
        angles_error = torch.mean(angles_error)
        self.writer.add_scalar(
            f'real_time-aj_pos_error_track', pos_error, info["step"]
        )
        self.writer.add_scalar(
            f'real_time-aj_rot_error_track', angles_error, info["step"]
        )
