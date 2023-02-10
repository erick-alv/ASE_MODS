import torch

from env.tasks.humanoid import compute_humanoid_reset
from env.tasks.humanoid_motion_load_and_reset import HumanoidMotionAndReset
from env.tasks.humanoid_view_motion import compute_view_motion_reset
from utils.motion_lib import MotionLib
from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil
import t1_rew
import t1_obs
import math


class HumanoidImitationVRH(HumanoidMotionAndReset):
    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        # This forces the code that is using torques as actions in the sim. It should be in config
        cfg["env"]["pdControl"] = False
        # Actually when parsing there should be a check that when using torques this value must be 1
        cfg["env"]["controlFrequencyInv"] = 1

        super().__init__(cfg=cfg,
                         sim_params=sim_params,
                         physics_engine=physics_engine,
                         device_type=device_type,
                         device_id=device_id,
                         headless=headless)

        num_motions = self._motion_lib.num_motions()
        self._motion_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        self._motion_ids = torch.remainder(self._motion_ids, num_motions)
        self._motions_start_time = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)
        self.feet_contact_forces = self.__get_feet_contact_force_sensor()
        self.prev_feet_contact_forces = torch.clone(self.feet_contact_forces)

    def __get_feet_contact_force_sensor(self):
        fcf = self.vec_sensor_tensor.view(self.num_envs, 2, 6)[:, :, :3] #the first 3 arguments correspond to the linear force
        return fcf

    # Overloaded methods from Humanoid

    def _setup_character_props(self, key_bodies):
        asset_file = self.cfg["env"]["asset"]["assetFileName"]
        num_key_bodies = len(key_bodies)
        device = self.cfg["args"].device + ':' + str(self.cfg["args"].device_id)

        if (asset_file == "mjcf/amp_humanoid_vrh.xml"):
            self._dof_body_ids = [1, 2, 4, 5, 8, 9, 12, 13, 14, 15, 16, 17]
            self._dof_offsets = [0, 3, 6, 9, 10, 13, 14, 17, 18, 21, 24, 25, 28]
            self._dof_obs_size = 72  # 12 bodies * 6; left from original code
            self._num_actions = 28
            self._num_obs = 287 + 162 + 1 # sim + (#vrh pieces * 9 * number frames to take) + scale
            self._rigid_body_vrh_indices = torch.tensor([3, 7, 11], dtype=torch.long, device=device)
            self._rigid_body_joints_indices = torch.arange(18, dtype=torch.long, device=device)
            for v in self._rigid_body_vrh_indices:
                self._rigid_body_joints_indices = self._rigid_body_joints_indices[self._rigid_body_joints_indices != v]
        elif (asset_file == "mjcf/amp_humanoid.xml"):
            self._dof_body_ids = [1, 2, 3, 4, 6, 7, 9, 10, 11, 12, 13, 14]
            self._dof_offsets = [0, 3, 6, 9, 10, 13, 14, 17, 18, 21, 24, 25, 28]
            self._dof_obs_size = 72
            self._num_actions = 28
            self._num_obs = 287 + 810 + 1
            self._rigid_body_vrh_indices = torch.arange(15, dtype=torch.long, device=device)
            self._rigid_body_joints_indices = torch.arange(15, dtype=torch.long, device=device)

        else:
            print("Unsupported character config file: {s}".format(asset_file))
            assert (False)

    def _compute_reward(self, actions):
        rb_pos_gt, rb_rot_gt, rb_vel_gt, \
            dof_pos_gt, dof_vel_gt = self._motion_lib.get_rb_state(self._motion_ids,
                                                                   self.progress_buf * self.dt + self._motions_start_time)
        feet_contact_forces = self.feet_contact_forces
        prev_feet_contact_forces = self.prev_feet_contact_forces
        self.rew_buf[:] = t1_rew.compute_reward(
            self._dof_pos, dof_pos_gt,
            self._dof_vel, dof_vel_gt,
            self._rigid_body_pos, rb_pos_gt,
            self._rigid_body_vel, rb_vel_gt,
            self._rigid_body_joints_indices,
            feet_contact_forces, prev_feet_contact_forces
        )
        return

    def _compute_reset(self):
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
        if len(self._reset_default_env_ids) > 0: #need to find new motion id for these envs
            new_motion_ids = self._motion_lib.sample_motions(len(self._reset_default_env_ids))
            self._motion_ids[self._reset_default_env_ids] = new_motion_ids
            self._motions_start_time[self._reset_default_env_ids] = 0
        if len(self._reset_ref_env_ids) > 0:#these already have a respective motion
            self._motion_ids[self._reset_ref_env_ids] = self._reset_ref_motion_ids
            self._motions_start_time[self._reset_ref_env_ids] = self._reset_ref_motion_times
        super()._reset_env_tensors(env_ids)

    def _compute_humanoid_obs(self, env_ids=None, num_steps_info=6):#todo make part of config and ttr of class


        rb_poses_gt_acc = []
        rb_rots_gt_acc = []
        for i in range(num_steps_info):
            rb_pos_gt, rb_rot_gt, \
                _, _, _ = self._motion_lib.get_rb_state(self._motion_ids,
                                                        (self.progress_buf + i) * self.dt + self._motions_start_time
                                                        )
            rb_poses_gt_acc.append(rb_pos_gt[:, self._rigid_body_vrh_indices, :])
            rb_rots_gt_acc.append(rb_rot_gt[:, self._rigid_body_vrh_indices, :])
        rb_poses_gt_acc = torch.cat(rb_poses_gt_acc, dim=1)
        rb_rots_gt_acc = torch.cat(rb_rots_gt_acc, dim=1)

        if (env_ids is None):
            body_pos = self._rigid_body_pos
            body_rot = self._rigid_body_rot
            body_vel = self._rigid_body_vel
            body_ang_vel = self._rigid_body_ang_vel
            dof_pos = self._dof_pos
            dof_vel = self._dof_vel
            feet_contact_forces = self.feet_contact_forces
            env_rb_poses_gt_acc = rb_poses_gt_acc
            env_rb_rots_gt_acc = rb_rots_gt_acc
            
        else:
            body_pos = self._rigid_body_pos[env_ids]
            body_rot = self._rigid_body_rot[env_ids]
            body_vel = self._rigid_body_vel[env_ids]
            body_ang_vel = self._rigid_body_ang_vel[env_ids]
            dof_pos = self._dof_pos[env_ids]
            dof_vel = self._dof_vel[env_ids]
            feet_contact_forces = self.feet_contact_forces[env_ids]
            env_rb_poses_gt_acc = rb_poses_gt_acc[env_ids]
            env_rb_rots_gt_acc = rb_rots_gt_acc[env_ids]

        obs = t1_obs.get_obs(body_pos, body_rot, body_vel, body_ang_vel,
                             self._rigid_body_joints_indices, dof_pos, dof_vel, feet_contact_forces,
                             env_rb_poses_gt_acc, env_rb_rots_gt_acc, self._rigid_body_vrh_indices)
        return obs

    def post_physics_step(self):
        super().post_physics_step()
        self.prev_feet_contact_forces[:] = torch.clone(self.feet_contact_forces)
    def _reset_envs(self, env_ids):
        super()._reset_envs(env_ids)
        if (len(env_ids) > 0):
            #It is important to do this after the reset of actors and call or refresh actors; so that the values in
            # the sim tensors are already updated
            reset_envs_feet_contact_forces = self.__get_feet_contact_force_sensor()[env_ids]
            self.prev_feet_contact_forces[env_ids] = torch.clone(reset_envs_feet_contact_forces)# at the beginning the prev contact force is the same todo it might be good to initialize them with zero since we really do not know what happens when force is resetted
        return

    # Extra methods for visualizing

    # def render(self, sync_frame_time=False):
    #     super().render(sync_frame_time)
    #
    #     if self.viewer:
    #         self.gym.clear_lines(self.viewer)
    #         self._visualize_bodies_transforms(self._rigid_body_vrh_indices)
    #         self._visualize_bodies_transforms(self._rigid_body_joints_indices, sphere_color=(0, 0, 1))
    #         feet_ids = torch.tensor([14, 17], dtype=torch.long, device=self.device)
    #         feet_pos = self._rigid_body_pos[:, feet_ids, :]
    #         self._visualize_force(feet_pos, self.feet_contact_forces)

    def _visualize_bodies_transforms(self, body_ids, sphere_color=None):
        positions = self._rigid_body_pos[:, body_ids, :]
        rotations = self._rigid_body_rot[:, body_ids, :]
        self._visualize_pose(positions, rotations, sphere_color)

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
            num_lines = force[i].shape[0]
            verts = np.zeros(shape=(num_lines, 6), dtype=np.float32)
            begins = force_beg_pos[i].cpu().numpy()
            ends = begins + (force[i].cpu().numpy())
            verts[:, :3] = begins  # assigning the begin
            verts[:, 3:] = ends  # assigning ends
            verts = verts.ravel()
            self.gym.add_lines(self.viewer, self.envs[i], num_lines, verts,
                               np.array([1.0, 0., 0.] * num_lines, dtype=np.float32))


#TODO until how many steps does the rl agent does and how should this affect loading the motions?
# load the motion always from beginning? or something else? And how to init the agent accoridngly?

@torch.jit.script
def compute_ref_motion_reset(reset_buf, motion_lengths, progress_buf, dt, motions_start_time):
    # type: (Tensor, Tensor, Tensor, float, Tensor) -> Tuple[Tensor, Tensor]
    terminated = torch.zeros_like(reset_buf)
    motion_times = progress_buf * dt + motions_start_time
    reset = torch.where(motion_times > motion_lengths, torch.ones_like(reset_buf), terminated)
    return reset, terminated