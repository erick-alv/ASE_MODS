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

import os

import yaml

from utils.config import set_np_formatting, set_seed, get_args, parse_sim_params, load_cfg
from utils.parse_task import parse_task
from utils.common_constants import DATE_TIME_FORMAT

from rl_games.algos_torch import players
from rl_games.algos_torch import torch_ext
from rl_games.algos_torch import model_builder
from rl_games.common import env_configurations, experiment, vecenv
from rl_games.common.algo_observer import AlgoObserver
from rl_games.torch_runner import Runner


from real_time.imitPoseState import ImitPoseStateThreadSafe
from real_time.utils import all_transforms, check_if_button_A_pressed
from real_time.mqtt import connect_mqtt, subscribe

import numpy as np
import copy
import torch
from datetime import datetime

from learning import amp_agent
from learning import amp_players
from learning import amp_models
from learning import amp_network_builder

from learning import ase_agent
from learning import ase_players
from learning import ase_models
from learning import ase_network_builder

from learning import hrl_agent
from learning import hrl_players
from learning import hrl_models
from learning import hrl_network_builder

from learning import common_agent
from learning import common_player
from learning import common_tester
from learning import common_real_time_player
from learning import common_network_builder

args = None
cfg = None
cfg_train = None

def create_rlgpu_env(**kwargs):
    use_horovod = cfg_train['params']['config'].get('multi_gpu', False)
    if use_horovod:
        import horovod.torch as hvd

        rank = hvd.rank()
        print("Horovod rank: ", rank)

        cfg_train['params']['seed'] = cfg_train['params']['seed'] + rank

        args.device = 'cuda'
        args.device_id = rank
        args.rl_device = 'cuda:' + str(rank)

        cfg['rank'] = rank
        cfg['rl_device'] = 'cuda:' + str(rank)

    sim_params = parse_sim_params(args, cfg, cfg_train)
    task, env = parse_task(args, cfg, cfg_train, sim_params)

    print('num_envs: {:d}'.format(env.num_envs))
    print('num_actions: {:d}'.format(env.num_actions))
    print('num_obs: {:d}'.format(env.num_obs))
    print('num_states: {:d}'.format(env.num_states))
    
    frames = kwargs.pop('frames', 1)
    if frames > 1:
        env = wrappers.FrameStack(env, frames, False)
    return env


class RLGPUAlgoObserver(AlgoObserver):
    def __init__(self, use_successes=True):
        self.use_successes = use_successes
        return

    def after_init(self, algo):
        self.algo = algo
        self.consecutive_successes = torch_ext.AverageMeter(1, self.algo.games_to_track).to(self.algo.ppo_device)
        self.writer = self.algo.writer
        return

    def process_infos(self, infos, done_indices):
        if isinstance(infos, dict):
            if (self.use_successes == False) and 'consecutive_successes' in infos:
                cons_successes = infos['consecutive_successes'].clone()
                self.consecutive_successes.update(cons_successes.to(self.algo.ppo_device))
            if self.use_successes and 'successes' in infos:
                successes = infos['successes'].clone()
                self.consecutive_successes.update(successes[done_indices].to(self.algo.ppo_device))
        return

    def after_clear_stats(self):
        self.mean_scores.clear()
        return

    def after_print_stats(self, frame, epoch_num, total_time):
        if self.consecutive_successes.current_size > 0:
            mean_con_successes = self.consecutive_successes.get_mean()
            self.writer.add_scalar('successes/consecutive_successes/mean', mean_con_successes, frame)
            self.writer.add_scalar('successes/consecutive_successes/iter', mean_con_successes, epoch_num)
            self.writer.add_scalar('successes/consecutive_successes/time', mean_con_successes, total_time)
        return


class RLGPUEnv(vecenv.IVecEnv):
    def __init__(self, config_name, num_actors, **kwargs):
        self.env = env_configurations.configurations[config_name]['env_creator'](**kwargs)
        self.use_global_obs = (self.env.num_states > 0)

        self.full_state = {}
        self.full_state["obs"] = self.reset()
        if self.use_global_obs:
            self.full_state["states"] = self.env.get_state()
        return

    def step(self, action):
        next_obs, reward, is_done, info = self.env.step(action)

        self.full_state["obs"] = next_obs
        if self.use_global_obs:
            self.full_state["states"] = self.env.get_state()
            return self.full_state, reward, is_done, info
        else:
            return self.full_state["obs"], reward, is_done, info

    def update_rew_weights(self, epoch_num):
        self.env.update_rew_weights(epoch_num)

    def reset(self, env_ids=None):
        self.full_state["obs"] = self.env.reset(env_ids)
        if self.use_global_obs:
            self.full_state["states"] = self.env.get_state()
            return self.full_state
        else:
            return self.full_state["obs"]

    def get_number_of_agents(self):
        return self.env.get_number_of_agents()

    def get_env_info(self):
        info = {}
        info['action_space'] = self.env.action_space
        info['observation_space'] = self.env.observation_space
        if hasattr(self.env, "amp_observation_space"):
            info['amp_observation_space'] = self.env.amp_observation_space

        if self.use_global_obs:
            info['state_space'] = self.env.state_space
            print(info['action_space'], info['observation_space'], info['state_space'])
        else:
            print(info['action_space'], info['observation_space'])

        return info


vecenv.register('RLGPU', lambda config_name, num_actors, **kwargs: RLGPUEnv(config_name, num_actors, **kwargs))
env_configurations.register('rlgpu', {
    'env_creator': lambda **kwargs: create_rlgpu_env(**kwargs),
    'vecenv_type': 'RLGPU'})

def build_alg_runner(algo_observer):
    runner = Runner(algo_observer)
    runner.algo_factory.register_builder('amp', lambda **kwargs : amp_agent.AMPAgent(**kwargs))
    runner.player_factory.register_builder('amp', lambda **kwargs : amp_players.AMPPlayerContinuous(**kwargs))
    runner.model_builder.model_factory.register_builder('amp', lambda network, **kwargs : amp_models.ModelAMPContinuous(network))
    runner.model_builder.network_factory.register_builder('amp', lambda **kwargs : amp_network_builder.AMPBuilder())
    
    runner.algo_factory.register_builder('ase', lambda **kwargs : ase_agent.ASEAgent(**kwargs))
    runner.player_factory.register_builder('ase', lambda **kwargs : ase_players.ASEPlayer(**kwargs))
    runner.model_builder.model_factory.register_builder('ase', lambda network, **kwargs : ase_models.ModelASEContinuous(network))
    runner.model_builder.network_factory.register_builder('ase', lambda **kwargs : ase_network_builder.ASEBuilder())
    
    runner.algo_factory.register_builder('hrl', lambda **kwargs : hrl_agent.HRLAgent(**kwargs))
    runner.player_factory.register_builder('hrl', lambda **kwargs : hrl_players.HRLPlayer(**kwargs))
    runner.model_builder.model_factory.register_builder('hrl', lambda network, **kwargs : hrl_models.ModelHRLContinuous(network))
    runner.model_builder.network_factory.register_builder('hrl', lambda **kwargs : hrl_network_builder.HRLBuilder())

    runner.algo_factory.register_builder('common', lambda **kwargs: common_agent.CommonAgent(**kwargs))
    #from rl_games.algos_torch.players import PpoPlayerContinuous
    #runner.player_factory.register_builder('common', lambda **kwargs: PpoPlayerContinuous(**kwargs))
    runner.player_factory.register_builder('common', lambda **kwargs : common_player.CommonPlayer(**kwargs))
    runner.player_factory.register_builder('common_test', lambda **kwargs: common_tester.CommonTester(**kwargs))
    runner.player_factory.register_builder('common_real_time', lambda **kwargs: common_real_time_player.CommonRealTimePlayer(**kwargs))

    runner.model_builder.network_factory.register_builder('common', lambda **kwargs: common_network_builder.CommonBuilder())
    
    return runner

def main():
    global args
    global cfg
    global cfg_train

    set_np_formatting()
    args = get_args()
    cfg, cfg_train, logdir = load_cfg(args)

    cfg_train['params']['seed'] = set_seed(cfg_train['params'].get("seed", -1), cfg_train['params'].get("torch_deterministic", False))

    if args.horovod:
        cfg_train['params']['config']['multi_gpu'] = args.horovod

    if args.horizon_length != -1:
        cfg_train['params']['config']['horizon_length'] = args.horizon_length

    if args.minibatch_size != -1:
        cfg_train['params']['config']['minibatch_size'] = args.minibatch_size
        
    if args.motion_file:
        cfg['env']['motion_file'] = args.motion_file


    # Create default directories for weights and statistics
    cfg_train['params']['config']['train_dir'] = args.output_path
    cfg_train['params']['config']['checkpoint'] = args.checkpoint

    cfg["env"]["real_time"] = args.real_time
    cfg["env"]["test"] = args.test
    cfg["env"]["play"] = args.play

    #creates full experiment name
    cfg_train['params']['config']['full_experiment_name'] = \
        cfg_train['params']['config']['name'] + datetime.now().strftime(DATE_TIME_FORMAT)

    vargs = vars(args)

    #creates the logdir
    if vargs["train"]:
        output_dir = vargs['output_path'] + cfg_train['params']['config']['full_experiment_name']
    elif vargs["test"]:
        checkpoint_path_els = vargs["checkpoint"].split(os.sep)
        if vargs["real_time"]:
            test_extra = "_test_results_real_time"
        else:
            test_extra = "_test_results"
        datetime_str = datetime.now().strftime(DATE_TIME_FORMAT)
        output_dir = os.path.join(checkpoint_path_els[0], checkpoint_path_els[1] + test_extra, datetime_str)
        cfg_train['params']['config']['test_results_dir'] = output_dir

    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    run_config_file = os.path.join(output_dir, "run_config.yaml")
    with open(run_config_file, 'w') as f:
        all_cfg = {
            'cfg': cfg,
            'cfg_train': cfg_train,
            'args': vargs

        }
        yaml.dump(all_cfg, f)


    try:

        #create the thread that will read the input
        if args.real_time:
            imitState = ImitPoseStateThreadSafe(cfg["env"]["imitParams"]["num_steps_track_info"])
            cfg["env"]["imitState"] = imitState

            # imitState_insert_func = lambda line: imitState.insert(line, transform_func=all_transforms, start_check_func=check_if_button_A_pressed)
            # asyncReadManager = AsyncInThreadManager()
            # asyncReadManager.submit_async(
            #     read_file(None, line_func=imitState_insert_func)
            # )

            def insert_msg_to_state(client, userdata, msg):
                #t = all_transforms(msg.payload.decode())
                #print(f"Received {t}")
                imitState.insert(msg.payload.decode(), transform_func=all_transforms,
                                 start_check_func=check_if_button_A_pressed)

            client = connect_mqtt('localhost', 1883)
            subscribe(client, "pico", insert_msg_to_state)
            client.loop_start()

        algo_observer = RLGPUAlgoObserver()

        runner = build_alg_runner(algo_observer)
        runner.load(cfg_train)
        runner.reset()
        runner.run(vargs)
    finally:
        if args.real_time:
            client.loop_stop(force=False)
            #asyncReadManager.stop_async()


    return

if __name__ == '__main__':
    main()
