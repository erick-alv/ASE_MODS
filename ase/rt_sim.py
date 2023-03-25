from isaacgym import gymapi, gymutil, gymtorch
import math
import torch
from ase.utils.motion_lib import MotionLib
import numpy as np
import t1_obs
import t1_rew
import asyncio
from contextlib import suppress
from reader import read_file
from imitPoseState import ImitPoseState

device = 'cuda:0'


def setup_sim():
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

    cam_pos = gymapi.Vec3(12.0, 0.0, 3)
    cam_target = gymapi.Vec3(8.0, 0.0, 0.0)
    gym.viewer_camera_look_at(viewer, None, cam_pos, cam_target)
    return gym, sim, viewer


def load_asset(gym, sim):
    asset_root = "ase/data/assets/"
    asset_file = "mjcf/amp_humanoid_vrh.xml"
    asset_options = gymapi.AssetOptions()
    asset_options.angular_damping = 0.01
    asset_options.max_angular_velocity = 100.0
    asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE  # gymapi.DOF_MODE_EFFORT #gymapi.DOF_MODE_NONE
    humanoid_asset = gym.load_asset(sim, asset_root, asset_file, asset_options)
    num_bodies = gym.get_asset_rigid_body_count(humanoid_asset)
    num_dofs = gym.get_asset_dof_count(humanoid_asset)
    return humanoid_asset, num_bodies, num_dofs


def setup_envs(gym, sim, num_envs, num_per_row, spacing, humanoid_asset, num_bodies):
    env_lower = gymapi.Vec3(-spacing, -spacing, 0.0)
    env_upper = gymapi.Vec3(spacing, spacing, spacing)
    # cache useful handles
    envs = []
    humanoid_handles = []
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

        start_pose.p = gymapi.Vec3(0.0, 0.0, 0.89)
        start_pose.p = gymapi.Vec3(0.2, 0.2, 0.89)
        start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        humanoid_handle = gym.create_actor(env, humanoid_asset, start_pose, "humanoid",
                                           col_group, col_filter, segmentation_id)

        for j in range(num_bodies):
            gym.set_rigid_body_color(env, humanoid_handle, j, gymapi.MESH_VISUAL, gymapi.Vec3(0.54, 0.85, 0.2))

        #gym.enable_actor_dof_force_sensors(env, humanoid_handle)
        humanoid_handles.append(humanoid_handle)
    return envs, humanoid_handles


def get_state_tensors(gym, sim, num_envs, num_bodies):
    rigid_body_state = gym.acquire_rigid_body_state_tensor(sim)
    gym.refresh_rigid_body_state_tensor(sim)
    _rigid_body_state = gymtorch.wrap_tensor(rigid_body_state)
    rigid_body_state_reshaped = _rigid_body_state.view(num_envs, num_bodies, 13)
    _rigid_body_pos = rigid_body_state_reshaped[:, :, 0:3]
    _rigid_body_rot = rigid_body_state_reshaped[:, :, 3:7]
    _rigid_body_vel = rigid_body_state_reshaped[:, :, 7:10]
    _rigid_body_ang_vel = rigid_body_state_reshaped[:, :, 10:13]
    return rigid_body_state, _rigid_body_pos, _rigid_body_rot, _rigid_body_vel, _rigid_body_ang_vel



def main():
    gym, sim, viewer = setup_sim()
    humanoid_asset, num_bodies, num_dofs = load_asset(gym, sim)

    # set up the env grid
    num_envs = 4
    num_per_row = 2
    spacing = 5
    envs, humanoid_handles = setup_envs(gym, sim, num_envs, num_per_row, spacing, humanoid_asset, num_bodies)

    # after setting up all environments
    gym.prepare_sim(sim)

    #getting states tensors
    prim_bs_tensor, _rigid_body_pos, _rigid_body_rot, _, _ = get_state_tensors(gym, sim, num_envs, num_bodies)

    async def main_loop(stop_event, shared_imitState):
        enable_viewer_sync = True
        while True and not gym.query_viewer_has_closed(viewer):

            # todo when implementing it should be read here before calling actions
            # if(await shared_imitState.is_ready()):
            #     val = await shared_imitState.get()
            #     print(f"From main_loop: {val}")

            actions = torch.zeros(num_envs * num_dofs, dtype=torch.float32, device=device).view(num_envs, num_dofs)
            gym.set_dof_actuation_force_tensor(sim, gymtorch.unwrap_tensor(actions))

            # physics step
            # render
            gym.fetch_results(sim, True)
            # MYEXTRA for displaying transforms
            gym.clear_lines(viewer)
            vrh_indices = [3, 7, 11]
            visualize_bodies_transforms(gym, viewer, envs, num_envs,
                                        _rigid_body_pos, _rigid_body_rot,
                                        body_ids=vrh_indices)
            if (await shared_imitState.is_ready()):
                val = await shared_imitState.get()
                val = val[0] #the first elment of the list (that is the first pose)
                #print(f"From main_loop: {val}")
                positions = torch.stack([val[0]]*num_envs, dim=0)
                rotations = torch.stack([val[1]]*num_envs, dim=0)
                visualize_pose(gym, viewer, envs, num_envs, positions, rotations, sphere_color=(1, 0, 0))


            # update the viewer
            if enable_viewer_sync:
                gym.step_graphics(sim)
                gym.draw_viewer(viewer, sim, True)
            else:
                gym.poll_viewer_events(viewer)
            # Wait for dt to elapse in real time.

            # step the physics
            gym.simulate(sim)

            # post physics
            gym.refresh_rigid_body_state_tensor(sim)
            await asyncio.sleep(0)

        stop_event.set()


    # the shared object between asyncio tasks
    imitState = ImitPoseState(1);

    import ast
    def tr_str(input_string):
        return ast.literal_eval(input_string[:-1])

    def to_torch(x, dtype=torch.float, device='cuda:0', requires_grad=False):
        return torch.tensor(x, dtype=dtype, device=device, requires_grad=requires_grad)

    def parse_pos_and_rot(float_torch_tensor):
        buttons_tensor = float_torch_tensor[-2:]
        pose_info = float_torch_tensor[:-2]
        poses_tensor = torch.reshape(pose_info, (3, 7))
        pos_tensor = poses_tensor[:, 0:3]
        rot_tensor = poses_tensor[:, 3:7]
        return pos_tensor, rot_tensor, buttons_tensor

    def transform_coord_system(input_el):
        pos_tensor, rot_tensor, buttons_tensor = input_el
        # right-handed rotation -90 around z
        tr_m1 = to_torch([
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0]
        ])

        # change dir of second axis (since isaac gym uses: positive y = left)
        tr_m2 = to_torch([
            [1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, 1.0]
        ])

        new_pos_tensor = []
        for i in range(pos_tensor.size()[0]):
            new_pos = torch.matmul(tr_m1, pos_tensor[i])
            new_pos = torch.matmul(tr_m2, new_pos)
            new_pos_tensor.append(new_pos)
        new_pos_tensor = torch.stack(new_pos_tensor)

        # swap axes
        tr_m = to_torch([
            [0.0, 0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ])

        new_rot_tensor = []
        for i in range(rot_tensor.size()[0]):
            new_rot = torch.matmul(tr_m, rot_tensor[i])
            # inverse since handedness changes; leave w the same
            new_rot[:3] *= -1
            # except for second (since isaac gym uses: positive y = left)
            new_rot[1] *= -1

            new_rot_tensor.append(new_rot)
        new_rot_tensor = torch.stack(new_rot_tensor)

        return new_pos_tensor, new_rot_tensor, buttons_tensor


    def all_transforms(input_info):
        return transform_coord_system(parse_pos_and_rot(to_torch(tr_str(input_info))))

    async def task_mngr(event_loop):
        stop_event = asyncio.Event()
        tasks = []
        tasks.append(
            event_loop.create_task(main_loop(stop_event, imitState))
        )
        tasks.append(
            event_loop.create_task(read_file(stop_event, lambda line: imitState.insert(line, transform_func=all_transforms)))
        )
        await stop_event.wait()
        for task in tasks:
            # cancel task
            task.cancel()
            # wait until cancelled
            with suppress(asyncio.CancelledError):
                await task

    loop = asyncio.get_event_loop()
    try:
        loop.run_until_complete(task_mngr(loop))
    finally:
        loop.run_until_complete(
            loop.shutdown_asyncgens())  # see: https://docs.python.org/3/library/asyncio-eventloop.html#asyncio.loop.shutdown_asyncgens
        loop.close()


    gym.destroy_viewer(viewer)
    gym.destroy_sim(sim)



def visualize_bodies_transforms(gym, viewer, envs, num_envs, _rigid_body_pos, _rigid_body_rot, body_ids, sphere_color=None):
    positions = _rigid_body_pos[:, body_ids, :]
    rotations = _rigid_body_rot[:, body_ids, :]
    visualize_pose(gym, viewer, envs, num_envs, positions, rotations, sphere_color)

#for visualization
def visualize_pose(gym, viewer, envs, num_envs, position, rotation, sphere_color=None):
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

if __name__ == "__main__":
    main()