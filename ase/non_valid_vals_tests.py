
from utils.env_obs_util import get_obs
from utils.env_rew_util import compute_reward
import torch


def random_tensor(shape, device, dtype, range, also_negative=True):
    rand_tensor = torch.rand(shape, device=device, dtype=dtype)
    if also_negative:
        rand_tensor = rand_tensor*2 - 1

    rand_tensor = rand_tensor*range
    return rand_tensor



def test_obs_function():
    N=100
    dev = "cuda:0"
    range = 10000.0

    body_pos = random_tensor((N, 18, 3), dev, torch.float, range)#(N, 18, 3)
    body_rot = random_tensor((N, 18, 4), dev, torch.float, range)#(N, 18, 4)
    body_vel = random_tensor((N, 18, 3), dev, torch.float, range)#(N, 18, 3)
    body_ang_vel = random_tensor((N, 18, 3), dev, torch.float, range)#(N, 18, 3)
    rigid_body_joints_indices = torch.tensor([0, 1, 2, 3,4,5,6, 7,8,9,10, 11, 12,13,14], device=dev, dtype=int) #(15) int within 18
    dof_pos = random_tensor((N, 28), dev, torch.float, range)#(N, 28)
    dof_vel = random_tensor((N, 38), dev, torch.float, range)#(N, 28)
    feet_contact_forces = random_tensor((N, 2, 3), dev, torch.float, range)#(N, 2, 3)
    env_rb_poses_gt_acc = random_tensor((N, 18, 3), dev, torch.float, range)#(N, 18, 3)
    env_rb_rots_gt_acc = random_tensor((N, 18, 4), dev, torch.float, range)#(N, 18, 4)
    imit_heights = random_tensor((N), dev, torch.float, range)#(N)
    humanoid_heights = random_tensor((N), dev, torch.float, range)#(N)
    
    obs = get_obs(body_pos, body_rot, body_vel, body_ang_vel, rigid_body_joints_indices, dof_pos, dof_vel,
                  feet_contact_forces, env_rb_poses_gt_acc, env_rb_rots_gt_acc, imit_heights, humanoid_heights)
    if torch.any(torch.isnan(obs)).item():
        print(obs)
        raise Exception("obs has nan value")
    if torch.any(torch.isinf(obs)).item():
        print(obs)
        raise Exception("obs has inf value")


def test_rew_function():
    N = 100
    dev = "cuda:0"
    range = 10000.0

    rigid_body_pos = random_tensor((N, 18, 3), dev, torch.float, range)  # (N, 18, 3)
    rigid_body_pos_gt = random_tensor((N, 18, 3), dev, torch.float, range)  # (N, 18, 3)
    rigid_body_vel = random_tensor((N, 18, 3), dev, torch.float, range)  # (N, 18, 3)
    rigid_body_vel_gt = random_tensor((N, 18, 3), dev, torch.float, range)  # (N, 18, 3)
    rigid_body_joints_indices = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14], device=dev,
                                             dtype=int)  # (15) int within 18
    dof_pos = random_tensor((N, 28), dev, torch.float, range)  # (N, 28)
    dof_pos_gt = random_tensor((N, 28), dev, torch.float, range)  # (N, 28)
    dof_vel = random_tensor((N, 38), dev, torch.float, range)  # (N, 28)
    dof_vel_gt = random_tensor((N, 38), dev, torch.float, range)  # (N, 28)
    feet_contact_forces = random_tensor((N, 2, 3), dev, torch.float, range)
    b_val = torch.randint(low=0, high=2, size=(1,))
    if b_val.item() == 0:
        prev_feet_contact_forces = random_tensor((N, 2, 3), dev, torch.float, range)# (N, 2, 3)
    else:
        prev_feet_contact_forces = torch.zeros_like(feet_contact_forces, device=dev)

    rew = compute_reward(
        dof_pos, dof_pos_gt,
        dof_vel, dof_vel_gt,
        rigid_body_pos, rigid_body_pos_gt,
        rigid_body_vel, rigid_body_vel_gt,
        rigid_body_joints_indices,
        feet_contact_forces, prev_feet_contact_forces,
        w_dof_pos=0.4, w_dof_vel=0.1, w_pos=0.2, w_vel=0.1, w_force=0.2,
        k_dof_pos=40.0, k_dof_vel=0.3, k_pos=6.0, k_vel=2.0, k_force=0.01)
    if torch.any(torch.isnan(rew)).item():
        print(rew)
        raise Exception("rew has nan value")
    if torch.any(torch.isinf(rew)).item():
        print(rew)
        raise Exception("rew has inf value")


if __name__ == "__main__":
    for i in range(1000):
        print(i)
        test_obs_function()
        test_rew_function()

    print("SUCCESS")
