from learning.common_player import CommonPlayer

import torch
from tensorboardX import SummaryWriter

import time
import os
from datetime import datetime
import re
from utils.motion_lib import MotionLib


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

        env_motion_lib :MotionLib = self.env_config["env"]["motion_lib"]
        num_motions = env_motion_lib.num_motions()
        for m_i in range(num_motions):
            if games_played >= n_games:#todo delete thi check
                break

            obs_dict = self.env_reset_with_motion(motion_ids=m_i)
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

                self._post_step(info)
                self._log_env_step(info, n)

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
                    if batch_size // self.num_agents == 1 or games_played >= n_games:
                        break

                done_indices = done_indices[:, 0]


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



    def _log_env_step(self, info, step):
        joints_indices = self.env_config["env"]["asset"]["jointsIndices"]
        joints_names = self.env_config["env"]["asset"]["jointsNames"]
        num_envs = info["body_pos"].shape[0]
        for n in range(num_envs):
            for i in range(len(joints_indices)):
                j_id = joints_indices[i]
                j_name = joints_names[i]
                self.writer.add_scalar(
                    f'env_{n}/{j_name}/position/x',
                    info['body_pos'][n, j_id, 0], step)
                self.writer.add_scalar(
                    f'env_{n}/{j_name}/position/y',
                    info['body_pos'][n, j_id, 1], step)
                self.writer.add_scalar(
                    f'env_{n}/{j_name}/position/z',
                    info['body_pos'][n, j_id, 2], step)
