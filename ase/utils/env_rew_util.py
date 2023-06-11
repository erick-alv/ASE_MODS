import torch
from isaacgym.torch_utils import quat_mul, quat_conjugate, quat_unit
from utils.torch_utils import quat_to_angle_axis


@torch.jit.script
def estimate_dof_dif(
        dof_pos, dof_pos_gt,
        dof_vel, dof_vel_gt):
    # type: (Tensor, Tensor, Tensor, Tensor) -> Tuple[Tensor, Tensor]

    dof_pos_dif = dof_pos - dof_pos_gt
    dof_pos_sqdif_sum = torch.sum(dof_pos_dif.pow(2), dim=-1)
    dof_vel_dif = dof_vel - dof_vel_gt
    dof_vel_sqdif_sum = torch.sum(dof_vel_dif.pow(2), dim=-1)
    return dof_pos_sqdif_sum, dof_vel_sqdif_sum


# @torch.jit.script
# def estimate_dof_angle_point_dif(
#         dof_pos, dof_pos_gt,
#         dof_vel, dof_vel_gt):
#     # type: (Tensor, Tensor, Tensor, Tensor) -> Tuple[Tensor, Tensor]
#
#     #  angles in radians and angle velocities in radian per second
#     #  transform into points in unit circle and estimate Euclidean distance
#     dof_pos_gt_cos = torch.cos(dof_pos_gt)
#     dof_pos_gt_sin = torch.sin(dof_pos_gt)
#     dof_pos_gt_ucp = torch.stack([dof_pos_gt_cos, dof_pos_gt_sin], dim=-1)
#     dof_pos_cos = torch.cos(dof_pos)
#     dof_pos_sin = torch.sin(dof_pos)
#     dof_pos_ucp = torch.stack([dof_pos_cos, dof_pos_sin], dim=-1)
#     dof_pos_ucp_dif = dof_pos_ucp - dof_pos_gt_ucp
#     dof_pos_norm = torch.linalg.norm(dof_pos_ucp_dif, dim=-1)
#     dof_pos_sqdif_sum = torch.sum(dof_pos_norm.pow(2), dim=-1)
#
#     dof_vel_dif = dof_vel - dof_vel_gt
#     dof_vel_sqdif_sum = torch.sum(dof_vel_dif.pow(2), dim=-1)
#     return dof_pos_sqdif_sum, dof_vel_sqdif_sum


@torch.jit.script
def estimate_rot_diff(rigid_body_rot, rigid_body_rot_gt,
        rigid_body_joints_indices):
    # type: (Tensor, Tensor, Tensor) -> Tensor
    rot = rigid_body_rot[:, rigid_body_joints_indices, :]
    rot_gt = rigid_body_rot_gt[:, rigid_body_joints_indices, :]


    or_shape = rot.shape
    # first we reshape since quaternion function expect shape of (N, 4)
    rot = rot.reshape(or_shape[0] * or_shape[1], or_shape[2])
    rot_gt = rot_gt.reshape(or_shape[0] * or_shape[1], or_shape[2])
    # quaternion difference
    q_delta = quat_mul(rot_gt, quat_conjugate(rot))
    angles_error, _ = quat_to_angle_axis(q_delta)
    angles_error = angles_error.reshape(or_shape[0], or_shape[1])
    angles_error_sq_sum = torch.sum(angles_error.pow(2), dim=-1)
    return angles_error_sq_sum


@torch.jit.script
def estimate_rigid_body_dif(rigid_body_pos, rigid_body_pos_gt,
        rigid_body_vel, rigid_body_vel_gt,
        rigid_body_joints_indices):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor) -> Tuple[Tensor, Tensor]
    rbj_pos = rigid_body_pos[:, rigid_body_joints_indices, :]
    rbj_pos_gt = rigid_body_pos_gt[:, rigid_body_joints_indices, :]
    rbj_vel = rigid_body_vel[:, rigid_body_joints_indices, :]
    rbj_vel_gt = rigid_body_vel_gt[:, rigid_body_joints_indices, :]

    rbj_pos_dif = rbj_pos - rbj_pos_gt
    rbj_pos_norm = torch.linalg.norm(rbj_pos_dif, dim=-1)
    rbj_pos_sqnorm_sum = torch.sum(rbj_pos_norm.pow(2), dim=-1)

    rbj_vel_dif = rbj_vel - rbj_vel_gt
    rbj_vel_norm = torch.linalg.norm(rbj_vel_dif, dim=-1)
    rbj_vel_sqnorm_sum = torch.sum(rbj_vel_norm.pow(2), dim=-1)
    return rbj_pos_sqnorm_sum, rbj_vel_sqnorm_sum


@torch.jit.script
def estimate_feet_force_term(feet_contact_forces, prev_feet_contact_forces):
    # type: (Tensor, Tensor) -> Tensor
    feet_vertical_force_dif = prev_feet_contact_forces[:, :, 2] - feet_contact_forces[:, :, 2]
    zero_comp = torch.zeros_like(feet_vertical_force_dif,
                                 dtype=feet_vertical_force_dif.dtype, device=feet_vertical_force_dif.device)
    feet_force_terms = torch.maximum(zero_comp, feet_vertical_force_dif)
    feet_force_terms_sum = torch.sum(feet_force_terms, dim=-1)
    return feet_force_terms_sum


@torch.jit.script
def penalize_fallen(computed_reward, rigid_body_pos, rigid_body_pos_gt, termination_heights, key_bodies_ids,
                    fall_penalty):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, float) -> Tensor
    joints_heights = rigid_body_pos[:, :, 2]
    has_fallen = joints_heights < termination_heights
    # ignore the keyBodies (feet and hands)
    has_fallen[:, key_bodies_ids] = False
    # ignore if reference motion requires that the joints goes below this point
    joints_heights_gt = rigid_body_pos_gt[:, :, 2]
    has_fallen_gt = joints_heights_gt < termination_heights
    has_fallen[has_fallen_gt] = False
    # penalize env where it a joint has fallen
    has_fallen = torch.any(has_fallen, dim=-1)
    computed_reward[has_fallen] = fall_penalty
    return computed_reward


@torch.jit.script
def estimate_reach_pos_dif(rigid_body_pos, rigid_body_pos_gt):
    # type: (Tensor, Tensor) -> Tensor
    #get root positions on the floor
    pos = rigid_body_pos[:, 0, :2]
    pos_gt = rigid_body_pos_gt[:, 0, :2]
    pos_dif = pos - pos_gt
    pos_norm = torch.linalg.norm(pos_dif, dim=-1)
    pos_sqnorm = pos_norm.pow(2)
    return pos_sqnorm


@torch.jit.script
def estimate_feet_vel_dif(rigid_body_vel, rigid_body_vel_gt, feet_bodies_ids):
    # type: (Tensor, Tensor, Tensor) -> Tensor
    #get feet velocity as magnitude of vector
    feet_vel_vec = rigid_body_vel[:, feet_bodies_ids, :]
    feet_vel = torch.linalg.norm(feet_vel_vec, dim=-1)
    feet_vel_vec_gt = rigid_body_vel_gt[:, feet_bodies_ids, :]
    feet_vel_gt = torch.linalg.norm(feet_vel_vec_gt, dim=-1)
    vel_dif = feet_vel - feet_vel_gt
    # calculates mask when the ground truth is moving:
    gt_moving_mask = feet_vel_gt > 0.09
    vel_dif[gt_moving_mask] = 0.0
    vel_dif_sq_sum = torch.sum(vel_dif.pow(2), dim=-1)

    return vel_dif_sq_sum

# using an extra penalty when it falls
@torch.jit.script
def compute_reward(
        dof_pos, dof_pos_gt,
        dof_vel, dof_vel_gt,
        rigid_body_pos, rigid_body_pos_gt,
        rigid_body_rot, rigid_body_rot_gt,
        rigid_body_vel, rigid_body_vel_gt,
        rigid_body_joints_indices,
        feet_contact_forces, prev_feet_contact_forces,
        feet_bodies_ids,
        termination_heights, key_bodies_ids, fall_penalty,
        w_dof_pos, w_dof_vel, w_pos, w_vel, w_force,
        k_dof_pos, k_dof_vel, k_pos, k_vel, k_force,
        w_extra1, w_extra2, w_extra3, k_extra1, k_extra2, k_extra3,
        use_penalty):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, bool) -> Tensor

    dof_pos_sqdif_sum, dof_vel_sqdif_sum = estimate_dof_dif(dof_pos, dof_pos_gt, dof_vel, dof_vel_gt)
    rbj_rot_sqnorm_sum = estimate_rot_diff(rigid_body_rot, rigid_body_rot_gt, rigid_body_joints_indices)

    rbj_pos_sqnorm_sum, rbj_vel_sqnorm_sum = estimate_rigid_body_dif(rigid_body_pos, rigid_body_pos_gt, rigid_body_vel,
                                                                     rigid_body_vel_gt, rigid_body_joints_indices)
    feet_force_terms_sum = estimate_feet_force_term(feet_contact_forces, prev_feet_contact_forces)

    reach_pos_dif = estimate_reach_pos_dif(rigid_body_pos, rigid_body_pos_gt)
    feet_vel_dif_sum = estimate_feet_vel_dif(rigid_body_vel, rigid_body_vel_gt, feet_bodies_ids)

    #r_dof_pos = w_dof_pos * torch.exp(-k_dof_pos * dof_pos_sqdif_sum)
    r_dof_pos = w_dof_pos * torch.exp(-k_dof_pos * rbj_rot_sqnorm_sum)
    r_dof_vel = w_dof_vel * torch.exp(-k_dof_vel * dof_vel_sqdif_sum)
    r_pos = w_pos * torch.exp(-k_pos * rbj_pos_sqnorm_sum)
    r_vel = w_vel * torch.exp(-k_vel * rbj_vel_sqnorm_sum)
    r_force = w_force * torch.exp(-k_force * feet_force_terms_sum)
    r_reach_pos = w_extra1 * torch.exp(-k_extra1 * reach_pos_dif)
    r_feet_vel = w_extra2 * torch.exp(-k_extra2 * feet_vel_dif_sum)

    total_reward = r_dof_pos + r_dof_vel + r_pos + r_vel + r_force + r_reach_pos + r_feet_vel

    if use_penalty:
        total_reward = penalize_fallen(total_reward, rigid_body_pos, rigid_body_pos_gt, termination_heights, key_bodies_ids,
                                       fall_penalty)
    return total_reward


@torch.jit.script
def compute_reward_dofr(
        dof_pos, dof_pos_gt,
        dof_vel, dof_vel_gt,
        rigid_body_pos, rigid_body_pos_gt,
        rigid_body_rot, rigid_body_rot_gt,
        rigid_body_vel, rigid_body_vel_gt,
        rigid_body_joints_indices,
        feet_contact_forces, prev_feet_contact_forces,
        feet_bodies_ids,
        termination_heights, key_bodies_ids, fall_penalty,
        w_dof_pos, w_dof_vel, w_pos, w_vel, w_force,
        k_dof_pos, k_dof_vel, k_pos, k_vel, k_force,
        w_extra1, w_extra2, w_extra3, k_extra1, k_extra2, k_extra3,
        use_penalty):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, bool) -> Tensor

    dof_pos_sqdif_sum, dof_vel_sqdif_sum = estimate_dof_dif(dof_pos, dof_pos_gt, dof_vel, dof_vel_gt)

    rbj_pos_sqnorm_sum, rbj_vel_sqnorm_sum = estimate_rigid_body_dif(rigid_body_pos, rigid_body_pos_gt, rigid_body_vel,
                                                                     rigid_body_vel_gt, rigid_body_joints_indices)
    feet_force_terms_sum = estimate_feet_force_term(feet_contact_forces, prev_feet_contact_forces)

    reach_pos_dif = estimate_reach_pos_dif(rigid_body_pos, rigid_body_pos_gt)
    feet_vel_dif_sum = estimate_feet_vel_dif(rigid_body_vel, rigid_body_vel_gt, feet_bodies_ids)



    r_dof_pos = w_dof_pos * torch.exp(-k_dof_pos * dof_pos_sqdif_sum)
    r_dof_vel = w_dof_vel * torch.exp(-k_dof_vel * dof_vel_sqdif_sum)
    r_pos = w_pos * torch.exp(-k_pos * rbj_pos_sqnorm_sum)
    r_vel = w_vel * torch.exp(-k_vel * rbj_vel_sqnorm_sum)
    r_force = w_force * torch.exp(-k_force * feet_force_terms_sum)
    r_reach_pos = w_extra1 * torch.exp(-k_extra1 * reach_pos_dif)
    r_feet_vel = w_extra2 * torch.exp(-k_extra2 * feet_vel_dif_sum)

    total_reward = r_dof_pos + r_dof_vel + r_pos + r_vel + r_force + r_reach_pos + r_feet_vel

    if use_penalty:
        total_reward = penalize_fallen(total_reward, rigid_body_pos, rigid_body_pos_gt, termination_heights, key_bodies_ids,
                                       fall_penalty)
    return total_reward