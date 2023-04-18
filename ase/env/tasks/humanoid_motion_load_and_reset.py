import torch
from isaacgym import gymapi
from isaacgym import gymtorch

from enum import Enum
from env.tasks.humanoid import Humanoid, dof_to_obs
from utils import gym_util
from utils.motion_lib import MotionLib, MultipleMotionLib
from isaacgym.torch_utils import *
from real_time.imitPoseState import ImitPoseStateThreadSafe
import time

from utils import torch_utils

class HumanoidMotionAndReset(Humanoid):
    class StateInit(Enum):
        Default = 0
        Start = 1
        Random = 2
        Hybrid = 3
        RealTime = 4

    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        if cfg["env"]["real_time"]:
            self._state_init = HumanoidMotionAndReset.StateInit.RealTime
        else:
            state_init = cfg["env"]["stateInit"]
            self._state_init = HumanoidMotionAndReset.StateInit[state_init]
            self._hybrid_init_prob = cfg["env"]["hybridInitProb"]
        self._reset_default_env_ids = []
        self._reset_ref_env_ids = []

        super().__init__(cfg=cfg,
                         sim_params=sim_params,
                         physics_engine=physics_engine,
                         device_type=device_type,
                         device_id=device_id,
                         headless=headless)

        if not self.cfg["env"]["real_time"]:
            motion_file = cfg['env']['motion_file']
            self.use_multiple_heights = cfg["env"]["asset"]["multipleHeights"]
            self._load_motion(motion_file)
            self.hard_reset_motion_ids = None
        else:
            self.imitState : ImitPoseStateThreadSafe = self.cfg["env"]["imitState"]
    
    def _load_motion(self, motion_file):
        assert (self._dof_offsets[-1] == self.num_dof)
        if self.use_multiple_heights:
            self._motion_lib = MultipleMotionLib(motion_file=motion_file,
                                         dof_body_ids=self._dof_body_ids,
                                         dof_offsets=self._dof_offsets,
                                         key_body_ids=self._key_body_ids.cpu().numpy(),
                                         device=self.device)
        else:
            self._motion_lib = MotionLib(motion_file=motion_file,
                                         dof_body_ids=self._dof_body_ids,
                                         dof_offsets=self._dof_offsets,
                                         key_body_ids=self._key_body_ids.cpu().numpy(),
                                         device=self.device)

        
    def _reset_envs(self, env_ids):
        self._reset_default_env_ids = []
        self._reset_ref_env_ids = []
        super()._reset_envs(env_ids)

    def _reset_actors(self, env_ids):
        if (self._state_init == HumanoidMotionAndReset.StateInit.Default):
            self._reset_default(env_ids)
        elif (self._state_init == HumanoidMotionAndReset.StateInit.Start
              or self._state_init == HumanoidMotionAndReset.StateInit.Random):
            self._reset_ref_state_init(env_ids)
        elif (self._state_init == HumanoidMotionAndReset.StateInit.Hybrid):
            self._reset_hybrid_state_init(env_ids)
        elif (self._state_init == HumanoidMotionAndReset.StateInit.RealTime):
            self._reset_real_time(env_ids)
        else:
            assert (False), "Unsupported state initialization strategy: {:s}".format(str(self._state_init))

    def _reset_default(self, env_ids):
        self._humanoid_root_states[env_ids] = self._initial_humanoid_root_states[env_ids]
        self._dof_pos[env_ids] = self._initial_dof_pos[env_ids]
        self._dof_vel[env_ids] = self._initial_dof_vel[env_ids]
        self._reset_default_env_ids = env_ids

    def _reset_ref_state_init(self, env_ids):
        num_envs = env_ids.shape[0]
        if self.hard_reset_motion_ids is not None:
            motion_ids = self.hard_reset_motion_ids
        else:
            motion_ids = self._motion_lib.sample_motions(num_envs)

        if (self._state_init == HumanoidMotionAndReset.StateInit.Random
                or self._state_init == HumanoidMotionAndReset.StateInit.Hybrid):
            motion_times = self._motion_lib.sample_time(motion_ids)
        elif (self._state_init == HumanoidMotionAndReset.StateInit.Start):
            motion_times = torch.zeros(num_envs, device=self.device)
        else:
            assert (False), "Unsupported state initialization strategy: {:s}".format(str(self._state_init))

        if self.use_multiple_heights:
            #we use the humanoid asset heights in order to reset the agent in a correct height
            motion_heights = self.humanoid_heights[env_ids]
            root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel, key_pos \
                = self._motion_lib.get_motion_state(motion_ids, motion_times, motion_heights)
        else:
            root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel, key_pos \
                = self._motion_lib.get_motion_state(motion_ids, motion_times)

        self._set_env_state(env_ids=env_ids,
                            root_pos=root_pos,
                            root_rot=root_rot,
                            dof_pos=dof_pos,
                            root_vel=root_vel,
                            root_ang_vel=root_ang_vel,
                            dof_vel=dof_vel)

        self._reset_ref_env_ids = env_ids
        self._reset_ref_motion_ids = motion_ids
        self._reset_ref_motion_times = motion_times
        return

    def _reset_hybrid_state_init(self, env_ids):
        num_envs = env_ids.shape[0]
        ref_probs = to_torch(np.array([self._hybrid_init_prob] * num_envs), device=self.device)
        ref_init_mask = torch.bernoulli(ref_probs) == 1.0

        ref_reset_ids = env_ids[ref_init_mask]
        if (len(ref_reset_ids) > 0):
            self._reset_ref_state_init(ref_reset_ids)

        default_reset_ids = env_ids[torch.logical_not(ref_init_mask)]
        if (len(default_reset_ids) > 0):
            self._reset_default(default_reset_ids)


        return

    def _reset_real_time(self, env_ids):
        #wait until user indicates the it is in the starting position
        while not (self.imitState.has_start_pose()):
            #print("Waiting for start pose")
            time.sleep(0.1)
        #first reset default since that is the start pose that we are expecting of the user
        self._reset_default(env_ids)
        #then adjusts to the current orientation and position
        start_pose = self.imitState.get_start_pose()
        # uses the HMD (index 0) as reference
        start_position_xy = start_pose[0][0][0:2]
        start_position_xy = torch.stack([start_position_xy]*self.num_envs, dim=0)
        start_rotation = start_pose[1][0]
        start_rotation = torch.stack([start_rotation]*self.num_envs, dim=0)

        # todo what to do with height
        self._humanoid_root_states[env_ids, 0:2] = start_position_xy[env_ids, :] #just x and y position
        self._humanoid_root_states[env_ids, 3:7] = start_rotation[env_ids, :]




    def _set_env_state(self, env_ids, root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel):
        self._humanoid_root_states[env_ids, 0:3] = root_pos
        self._humanoid_root_states[env_ids, 3:7] = root_rot
        self._humanoid_root_states[env_ids, 7:10] = root_vel
        self._humanoid_root_states[env_ids, 10:13] = root_ang_vel

        self._dof_pos[env_ids] = dof_pos
        self._dof_vel[env_ids] = dof_vel