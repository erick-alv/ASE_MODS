from isaacgym import gymapi, gymutil, gymtorch
import math
import torch
from ase.utils.motion_lib import MotionLib
import numpy as np
from utils import env_obs_util, env_rew_util

device = 'cuda:0'


def to_torch(x, dtype=torch.float, device='cuda:0', requires_grad=False):
    return torch.tensor(x, dtype=dtype, device=device, requires_grad=requires_grad)


# parse arguments
args = gymutil.parse_arguments(
    description="Joint monkey: Animate degree-of-freedom ranges",
    custom_parameters=[
        {"name": "--speed_scale", "type": float, "default": 1.0, "help": "Animation speed scale"},
        {"name": "--show_axis", "action": "store_true", "help": "Visualize DOF axis"}
    ])

# initialize gym
gym = gymapi.acquire_gym()

# configure sim
sim_params = gymapi.SimParams()
sim_params.dt = dt = 1.0 / 60.0
sim_params.up_axis = gymapi.UP_AXIS_Z
sim_params.gravity.x = 0
sim_params.gravity.y = 0
sim_params.gravity.z = -9.81
# PHYSX configuration
sim_params.physx.solver_type = 1
sim_params.physx.num_position_iterations = 4
sim_params.physx.num_velocity_iterations = 0
sim_params.physx.num_threads = 4
sim_params.physx.use_gpu = args.use_gpu
# when True, GPU tensors are used and returned by GYM
sim_params.use_gpu_pipeline = args.use_gpu_pipeline

sim = gym.create_sim(args.compute_device_id, args.graphics_device_id, args.physics_engine, sim_params)

# add ground plane
plane_params = gymapi.PlaneParams()
plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
gym.add_ground(sim, plane_params)

# create viewer
viewer = gym.create_viewer(sim, gymapi.CameraProperties())
if viewer is None:
    print("*** Failed to create viewer")
    quit()

# position the camera
cam_pos = gymapi.Vec3(12.0, 0.0, 3)
cam_target = gymapi.Vec3(8.0, 0.0, 0.0)
gym.viewer_camera_look_at(viewer, None, cam_pos, cam_target)

# load asset
asset_root = "ase/data/assets/"
# asset_file = "mjcf/amp_humanoid.xml"
# asset_file = "mjcf/amp_humanoid_sword_shield.xml"
asset_file = "mjcf/amp_humanoid_vrh.xml"
#asset_file = "mjcf/amp_humanoid_vrh(backup).xml"
asset_options = gymapi.AssetOptions()
asset_options.angular_damping = 0.01
asset_options.max_angular_velocity = 100.0
asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE  # gymapi.DOF_MODE_EFFORT #gymapi.DOF_MODE_NONE
humanoid_asset = gym.load_asset(sim, asset_root, asset_file, asset_options)

num_bodies = gym.get_asset_rigid_body_count(humanoid_asset)
num_dofs = gym.get_asset_dof_count(humanoid_asset)
actuator_props = gym.get_asset_actuator_properties(humanoid_asset)
motor_efforts = [prop.motor_effort for prop in actuator_props]
max_motor_effort = max(motor_efforts)
motor_efforts = to_torch(motor_efforts, device=device)
power_scale = 1

# set up the env grid
num_envs = 4
num_per_row = 2
spacing = 5
env_lower = gymapi.Vec3(-spacing, -spacing, 0.0)
env_upper = gymapi.Vec3(spacing, spacing, spacing)
# cache useful handles
envs = []
humanoid_handles = []
dof_limits_lower = []
dof_limits_upper = []
print("Creating %d environments" % num_envs)
for i in range(num_envs):
    # create env
    env = gym.create_env(sim, env_lower, env_upper, num_per_row)
    envs.append(env)
    # create actor
    col_group = i
    col_filter = 0  # 1: disables self collisions, 0: with collisions
    segmentation_id = 0

    start_pose = gymapi.Transform()

    start_pose.p = gymapi.Vec3(0.0, 0.0, 1.06)
    start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

    humanoid_handle = gym.create_actor(env, humanoid_asset, start_pose, "humanoid",
                                       col_group, col_filter, segmentation_id)

    for j in range(num_bodies):
        gym.set_rigid_body_color(env, humanoid_handle, j, gymapi.MESH_VISUAL, gymapi.Vec3(0.54, 0.85, 0.2))

    gym.enable_actor_dof_force_sensors(env, humanoid_handle)
    humanoid_handles.append(humanoid_handle)

dof_prop = gym.get_actor_dof_properties(envs[0], humanoid_handles[0])
for j in range(num_dofs):
    if dof_prop['lower'][j] > dof_prop['upper'][j]:
        dof_limits_lower.append(dof_prop['upper'][j])
        dof_limits_upper.append(dof_prop['lower'][j])
    else:
        dof_limits_lower.append(dof_prop['lower'][j])
        dof_limits_upper.append(dof_prop['upper'][j])

dof_limits_lower = to_torch(dof_limits_lower, device=device)
dof_limits_upper = to_torch(dof_limits_upper, device=device)
# after setting up all environments
gym.prepare_sim(sim)

# Getting the tensors handles
actor_root_state = gym.acquire_actor_root_state_tensor(sim)
gym.refresh_actor_root_state_tensor(sim)
root_state = gymtorch.wrap_tensor(actor_root_state)
initial_root_state = root_state.clone()
initial_root_state[:, 7:13] = 0
root_pos = root_state[:, 0:3]
root_rot = root_state[:, 3:7]
root_vel = root_state[:, 7:10]
root_ang_vel = root_state[:, 10:13]

dof_state_tensor = gym.acquire_dof_state_tensor(sim)
gym.refresh_dof_state_tensor(sim)
dof_state = gymtorch.wrap_tensor(dof_state_tensor)
dof_pos = dof_state.view(num_envs, num_dofs, 2)[..., 0]
dof_vel = dof_state.view(num_envs, num_dofs, 2)[..., 1]
initial_dof_pos = torch.zeros_like(dof_pos, device=device, dtype=torch.float)
zero_tensor = torch.tensor([0.0], device=device)
initial_dof_pos = torch.where(dof_limits_lower > zero_tensor, dof_limits_lower,
                              torch.where(dof_limits_upper < zero_tensor, dof_limits_upper, initial_dof_pos))
initial_dof_vel = torch.zeros_like(dof_vel, device=device, dtype=torch.float)

rigid_body_state = gym.acquire_rigid_body_state_tensor(sim)
gym.refresh_rigid_body_state_tensor(sim)
_rigid_body_state = gymtorch.wrap_tensor(rigid_body_state)
rigid_body_state_reshaped = _rigid_body_state.view(num_envs, num_bodies, 13)
_rigid_body_pos = rigid_body_state_reshaped[:, :, 0:3]
_rigid_body_rot = rigid_body_state_reshaped[:, :, 3:7]
_rigid_body_vel = rigid_body_state_reshaped[:, :, 7:10]
_rigid_body_ang_vel = rigid_body_state_reshaped[:, :, 10:13]

height = _rigid_body_pos[:, 3, 2] - _rigid_body_pos[:, 17, 2]
print(height)

_rigid_body_track_indices = torch.tensor([3, 7, 11], dtype=torch.long, device=device)
_rigid_body_joints_indices = torch.arange(num_bodies, dtype=torch.long, device=device)
for v in _rigid_body_track_indices:
    _rigid_body_joints_indices = _rigid_body_joints_indices[_rigid_body_joints_indices != v]

dof_force_tensor = gym.acquire_dof_force_tensor(sim)
dof_force_tensor = gymtorch.wrap_tensor(dof_force_tensor).view(num_envs, num_dofs)

contact_force_tensor = gym.acquire_net_contact_force_tensor(sim)
gym.refresh_net_contact_force_tensor(sim)
contact_force_tensor = gymtorch.wrap_tensor(contact_force_tensor)
_contact_forces = contact_force_tensor.view(num_envs, num_bodies, 3)
feet_ids = torch.tensor([14, 17], dtype=torch.long, device=device)
feet_contact_forces = _contact_forces[:, feet_ids, :]  # MY


def _set_env_state(env_ids, arg_root_pos, arg_root_rot, arg_dof_pos, arg_root_vel, arg_root_ang_vel, arg_dof_vel):
    root_pos[env_ids] = arg_root_pos
    root_rot[env_ids] = arg_root_rot
    root_vel[env_ids] = arg_root_vel
    root_ang_vel[env_ids] = arg_root_ang_vel

    dof_pos[env_ids] = arg_dof_pos
    dof_vel[env_ids] = arg_dof_vel
    return


# def _set_env_state2(env_ids, arg_rb_pos, arg_rb_rot, arg_dof_pos, arg_rb_vel, arg_rb_ang_vel, arg_dof_vel, rb_indices):
#     _rigid_body_pos[env_ids][:, rb_indices, :] = arg_rb_pos
#     _rigid_body_rot[env_ids][:, rb_indices, :] = arg_rb_rot
#     _rigid_body_vel[env_ids][:, rb_indices, :] = arg_rb_vel
#     _rigid_body_ang_vel[env_ids][:, rb_indices, :] = arg_rb_ang_vel
#
#     dof_pos[env_ids] = arg_dof_pos
#     dof_vel[env_ids] = arg_dof_vel
#     return
#
#
# def _set_env_state3(env_ids, arg_rb_pos, arg_rb_rot, arg_rb_vel, arg_rb_ang_vel, rb_indices):
#     _rigid_body_pos[env_ids][:, rb_indices, :] = arg_rb_pos
#     _rigid_body_rot[env_ids][:, rb_indices, :] = arg_rb_rot
#     _rigid_body_vel[env_ids][:, rb_indices, :] = arg_rb_vel
#     _rigid_body_ang_vel[env_ids][:, rb_indices, :] = arg_rb_ang_vel
#     return


##loading motion capture data
if (asset_file == "mjcf/amp_humanoid.xml"):
    _dof_body_ids = [1, 2, 3, 4, 6, 7, 9, 10, 11, 12, 13, 14]
    _key_body_ids = [5, 8, 11, 14]  # ids of extremities
    _dof_offsets = [0, 3, 6, 9, 10, 13, 14, 17, 18, 21, 24, 25, 28]
elif (asset_file == "mjcf/amp_humanoid_vrh.xml" or asset_file == "mjcf/amp_humanoid_vrh(backup).xml"):
    _dof_body_ids = [1, 2, 4, 5, 8, 9, 12, 13, 14, 15, 16, 17]
    _key_body_ids = [6, 10, 14, 17]  # ids of extremities ends
    _dof_offsets = [0, 3, 6, 9, 10, 13, 14, 17, 18, 21, 24, 25, 28]

if (asset_file == "mjcf/amp_humanoid.xml"):
    motion_file = 'ase/data/motions/without_vrh/01_01_cmu_amp.npy'
    # motion_file = 'ase/data/motions/without_vrh/amp_humanoid_walk.npy'
    # motion_file = 'ase/data/motions/without_vrh/0007_Crawling001_amp.npy'
    #motion_file = 'ase/data/motions/without_vrh/sfu_amp/0008_Walking001_amp.npy'
elif (asset_file == "mjcf/amp_humanoid_vrh.xml" or asset_file == "mjcf/amp_humanoid_vrh(backup).xml"):
    #motion_file = 'ase/data/motions/cmu_temp_retargeted/01_01_cmu_amp.npy'
    #motion_file = 'ase/data/motions/cmu_temp_retargeted/49_08_cmu_amp.npy'
    motion_file = 'ase/data/motions/cmu_temp_retargeted/07_01_cmu_amp.npy'
    #motion_file = 'ase/data/motions/cmu_temp_retargeted/09_11_cmu_amp.npy'
    #motion_file = 'ase/data/motions/cmu_temp_retargeted/08_02_cmu_amp.npy'

    #motion_file = 'ase/data/motions/sfu_temp_retargeted/0005_Jogging001_amp.npy'
    # motion_file = 'ase/data/motions/sfu_temp_retargeted/0005_Walking001_amp.npy'
    # motion_file = 'ase/data/motions/sfu_temp_retargeted/0007_Balance001_amp.npy'
    # motion_file = 'ase/data/motions/sfu_temp_retargeted/0007_Walking001_amp.npy'
    # motion_file = 'ase/data/motions/sfu_temp_retargeted/0008_Walking001_amp.npy'
    # motion_file = 'ase/data/motions/sfu_temp_retargeted/0018_Walking001_amp.npy'

    #motion_file = "ase/data/motions/lafan_temp_retargeted/aiming1_subject1_amp.npy"
    #motion_file = "ase/data/motions/lafan_temp_retargeted/aiming1_subject4_amp.npy"
    #motion_file = "ase/data/motions/lafan_temp_retargeted/aiming2_subject2_amp.npy"
    #motion_file = "ase/data/motions/lafan_temp_retargeted/aiming2_subject3_amp.npy"
    #motion_file = "ase/data/motions/lafan_temp_retargeted/aiming2_subject5_amp.npy"
    #motion_file = "ase/data/motions/lafan_temp_retargeted/dance1_subject2_amp.npy"
    # motion_file = "ase/data/motions/lafan_temp_retargeted/dance2_subject4_amp.npy"
    #motion_file = "ase/data/motions/lafan_temp_retargeted/fallAndGetUp1_subject4_amp.npy"
    #motion_file = "ase/data/motions/lafan_temp_retargeted/fallAndGetUp1_subject5_amp.npy"
    #motion_file = "ase/data/motions/lafan_temp_retargeted/fallAndGetUp2_subject2_amp.npy"
    #motion_file = "ase/data/motions/lafan_temp_retargeted/obstacles1_subject1_amp.npy"
    # motion_file = "ase/data/motions/lafan_temp_retargeted/obstacles1_subject2_amp.npy"
    # motion_file = "ase/data/motions/lafan_temp_retargeted/obstacles1_subject5_amp.npy"
    # motion_file = "ase/data/motions/lafan_temp_retargeted/obstacles3_subject3_amp.npy"
    #motion_file = "ase/data/motions/lafan_temp_retargeted/obstacles3_subject4_amp.npy"
    #motion_file = "ase/poselib/data/cmu_motions_retargeted/conversation/19_10_amp.npy"

    # motion_file = "ase/data/motions/zeggs_temp_retargeted/conversation/026_Angry_0_x_1_0_amp.npy"
    # motion_file = "ase/data/motions/zeggs_temp_retargeted/conversation/001_Neutral_0_mirror_x_1_0_amp.npy"
    # motion_file = "ase/data/motions/zeggs_temp_retargeted/conversation/002_Neutral_1_x_1_0_amp.npy"
    # motion_file = "ase/data/motions/zeggs_temp_retargeted/conversation/012_Happy_1_x_1_0_amp.npy"

_motion_lib = MotionLib(motion_file=motion_file,
                        dof_body_ids=_dof_body_ids,
                        dof_offsets=_dof_offsets,
                        key_body_ids=_key_body_ids,
                        device=device)

num_motions = _motion_lib.num_motions()
_motion_ids = torch.arange(num_envs, device=device, dtype=torch.long)
_motion_ids = torch.remainder(_motion_ids, num_motions)


def _motion_sync(progress_buf, _motion_dt):
    motion_ids = _motion_ids
    motion_times = progress_buf * _motion_dt

    l_root_pos, l_root_rot, l_dof_pos, l_root_vel, l_root_ang_vel, l_dof_vel, l_key_pos \
        = _motion_lib.get_motion_state(motion_ids, motion_times)

    l_root_vel = torch.zeros_like(l_root_vel)
    l_root_ang_vel = torch.zeros_like(l_root_ang_vel)
    l_dof_vel = torch.zeros_like(l_dof_vel)

    env_ids = torch.arange(num_envs, dtype=torch.long, device=device)

    _set_env_state(env_ids=env_ids,
                   arg_root_pos=l_root_pos,
                   arg_root_rot=l_root_rot,
                   arg_dof_pos=l_dof_pos,
                   arg_root_vel=l_root_vel,
                   arg_root_ang_vel=l_root_ang_vel,
                   arg_dof_vel=l_dof_vel)

    _humanoid_actor_ids = 1 * torch.arange(num_envs, device=device, dtype=torch.int32)
    env_ids_int32 = _humanoid_actor_ids[env_ids]
    gym.set_actor_root_state_tensor_indexed(sim, gymtorch.unwrap_tensor(root_state),
                                            gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
    gym.set_dof_state_tensor_indexed(sim, gymtorch.unwrap_tensor(dof_state),
                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))


    return


# MYEXTRA for displaying transforms
# indices for headset, right_controller, left_controller
# h_handle = gym.find_actor_rigid_body_handle(envs[0], humanoid_handles[0], "headset")

def visualize_bodies_transforms(body_ids, sphere_color=None):
    positions = _rigid_body_pos[:, body_ids, :]
    rotations = _rigid_body_rot[:, body_ids, :]
    visualize_pose(positions, rotations, sphere_color)


def visualize_pose(position, rotation, sphere_color=None):
    # axes and sphere for transform
    axes_geom = gymutil.AxesGeometry(0.1)
    # Create a wireframe sphere
    sphere_rot = gymapi.Quat.from_euler_zyx(0.5 * math.pi, 0, 0)
    sphere_pose = gymapi.Transform(r=sphere_rot)
    if sphere_color is None:
        sphere_color = (1, 1, 0)
    sphere_geom = gymutil.WireframeSphereGeometry(0.02, 12, 12, sphere_pose, color=sphere_color)

    for i in range(num_envs):
        if position.ndim == 3:
            for j in range(position.shape[1]):
                pv = gymapi.Vec3(position[i, j, 0], position[i, j, 1], position[i, j, 2])
                pq = gymapi.Quat(rotation[i, j, 0], rotation[i, j, 1], rotation[i, j, 2], rotation[i, j, 3])
                pose = gymapi.Transform(pv, pq)
                gymutil.draw_lines(axes_geom, gym, viewer, envs[i], pose)
                gymutil.draw_lines(sphere_geom, gym, viewer, envs[i], pose)
        else:  # just 2D env and one single pos
            pv = gymapi.Vec3(position[i, 0], position[i, 1], position[i, 2])
            pq = gymapi.Quat(rotation[i, 0], rotation[i, 1], rotation[i, 2], rotation[i, 3])
            pose = gymapi.Transform(pv, pq)
            gymutil.draw_lines(axes_geom, gym, viewer, envs[i], pose)
            gymutil.draw_lines(sphere_geom, gym, viewer, envs[i], pose)


def visualize_force(force_beg_pos, force):
    for i in range(num_envs):
        num_lines = force[i].shape[0]
        verts = np.zeros(shape=(num_lines, 6), dtype=np.float32)
        begins = force_beg_pos[i].cpu().numpy()
        ends = begins + (force[i].cpu().numpy())
        verts[:, :3] = begins  # assigning the begin
        verts[:, 3:] = ends  # assigning ends
        verts = verts.ravel()
        gym.add_lines(viewer, envs[i], num_lines, verts, np.array([1.0, 0., 0.] * num_lines, dtype=np.float32))


########### MYEXTRA END for displaying transforms


##### MYEXTRA for checking
def check_poses(progress_buf, _motion_dt):
    hh_motion_ids = _motion_ids
    hh_motion_times = progress_buf * _motion_dt

    mlib_pos, mlib_rot, _, _, _ = _motion_lib.get_rb_state(hh_motion_ids, hh_motion_times)

    mlib_pos = mlib_pos[:, _rigid_body_joints_indices, :]
    mlib_rot = mlib_rot[:, _rigid_body_joints_indices, :]

    current_pos = _rigid_body_pos[:, _rigid_body_joints_indices, :]
    current_rot = _rigid_body_rot[:, _rigid_body_joints_indices, :]

    print(torch.all(torch.all(torch.isclose(current_pos, mlib_pos, atol=1e-1), dim=-1), dim=-1))
    # print(
    #     "*********************************************************\n*********************************************************")
    # print(torch.all(torch.isclose(current_rot, mlib_rot, atol=1e-1)))


def compare_root_pos_rb_root_pos():
    current_root_pos = root_pos
    current_root_rot = root_rot

    current_rb_pos = _rigid_body_pos[:, _rigid_body_joints_indices, :]
    current_rb_rot = _rigid_body_rot[:, _rigid_body_joints_indices, :]
    print(torch.isclose(current_root_pos, current_rb_pos[:, 0, :]))
    print(
        "*********************************************************\n*********************************************************")
    print(torch.isclose(current_root_rot, current_rb_rot[:, 0, :]))


##### MYEXTRA END for checking


progress_buf = torch.zeros(num_envs, device=device, dtype=torch.long)
it = 0
enable_viewer_sync = True
prev_feet_contact_forces = None
while it < 1000000 and not gym.query_viewer_has_closed(viewer):
    _motion_sync(progress_buf, dt)
    # pre physics
    # generating random action
    # actions = 1.0 - 2.0 * torch.rand(num_envs * num_dofs, dtype=torch.float32, device="cuda:0").view(num_envs, num_dofs)
    actions = torch.zeros(num_envs * num_dofs, dtype=torch.float32, device=device).view(num_envs, num_dofs)
    # make that env 0 does not have extra action
    # actions[0, :] = 0

    actions = torch.clamp(actions, -1, 1)
    ff = actions * motor_efforts.unsqueeze(0) * power_scale
    ff_tensor = gymtorch.unwrap_tensor(ff)
    gym.set_dof_actuation_force_tensor(sim, ff_tensor)

    # physics step
    # render
    gym.fetch_results(sim, True)
    # MYEXTRA for displaying transforms
    gym.clear_lines(viewer)
    track_indices = [3, 7, 11]
    visualize_bodies_transforms(track_indices)
    visualize_bodies_transforms(_rigid_body_joints_indices, sphere_color=(0, 0, 1))

    hh_motion_ids = _motion_ids
    hh_motion_times = progress_buf * dt
    mlib_pos, mlib_rot, _, _, _ = _motion_lib.get_rb_state(hh_motion_ids, hh_motion_times)
    visualize_pose(mlib_pos, mlib_rot, sphere_color=(1, 0, 0))

    # sframe_pos, sframe_heading_rot = env_obs_util._estimate_sframe(_rigid_body_pos, _rigid_body_rot)
    # visualize_pose(sframe_pos, sframe_heading_rot)
    feet_pos = _rigid_body_pos[:, feet_ids, :]
    visualize_force(feet_pos, feet_contact_forces)

    ########### MYEXTRA END for displaying transforms
    # update the viewer
    if enable_viewer_sync:
        gym.step_graphics(sim)
        gym.draw_viewer(viewer, sim, True)
    else:
        gym.poll_viewer_events(viewer)
    # Wait for dt to elapse in real time.
    # This synchronizes the physics simulation with the rendering rate.
    # gym.sync_frame_time(sim)
    # step the physics
    gym.simulate(sim)

    # post physics
    progress_buf += 1

    gym.refresh_dof_state_tensor(sim)
    gym.refresh_actor_root_state_tensor(sim)
    gym.refresh_rigid_body_state_tensor(sim)
    gym.refresh_net_contact_force_tensor(sim)
    # gym.refresh_force_sensor_tensor(sim)
    gym.refresh_dof_force_tensor(sim)

    rb_poses_gt_acc = []
    rb_rots_gt_acc = []
    for i in range(6):
        rb_pos_gt, rb_rot_gt, _, _, _ = _motion_lib.get_rb_state(_motion_ids, (progress_buf + i) * dt)
        rb_poses_gt_acc.append(rb_pos_gt[:, _rigid_body_track_indices, :])
        rb_rots_gt_acc.append(rb_rot_gt[:, _rigid_body_track_indices, :])
    rb_poses_gt_acc = torch.cat(rb_poses_gt_acc, dim=1)
    rb_rots_gt_acc = torch.cat(rb_rots_gt_acc, dim=1)

    obs = env_obs_util.get_obs(_rigid_body_pos, _rigid_body_rot, _rigid_body_vel, _rigid_body_ang_vel,
                         _rigid_body_joints_indices, dof_pos, dof_vel, feet_contact_forces,
                         rb_poses_gt_acc, rb_rots_gt_acc)


    rb_pos_gt, rb_rot_gt, rb_vel_gt, \
        dof_pos_gt, dof_vel_gt = _motion_lib.get_rb_state(_motion_ids, progress_buf * dt)
    if prev_feet_contact_forces is None:
        prev_feet_contact_forces = feet_contact_forces.clone()
    reward = env_rew_util.compute_reward(
        dof_pos, dof_pos_gt,
        dof_vel, dof_vel_gt,
        _rigid_body_pos, rb_pos_gt,
        _rigid_body_vel, rb_vel_gt,
        _rigid_body_joints_indices,
        feet_contact_forces, prev_feet_contact_forces,
        w_dof_pos=0.4, w_dof_vel=0.1, w_pos=0.2, w_vel=0.1, w_force=0.2,
        k_dof_pos=40.0, k_dof_vel=0.3, k_pos=6.0, k_vel=2.0, k_force=0.01)

    print(reward)

    print(
        "_________________________________________________________\n_________________________________________________________")
    # compare_root_pos_rb_root_pos()
    check_poses(progress_buf, dt)

    it += 0.001
    prev_feet_contact_forces = feet_contact_forces.clone()

gym.destroy_viewer(viewer)
gym.destroy_sim(sim)
