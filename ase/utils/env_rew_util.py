import torch

#same reward as in questsim
@torch.jit.script
def compute_reward(
        dof_pos, dof_pos_gt,
        dof_vel, dof_vel_gt,
        rigid_body_pos, rigid_body_pos_gt,
        rigid_body_vel, rigid_body_vel_gt,
        rigid_body_joints_indices,
        feet_contact_forces, prev_feet_contact_forces,
        termination_heights, key_bodies_ids, fall_penalty,
        w_dof_pos, w_dof_vel, w_pos, w_vel, w_force,
        k_dof_pos, k_dof_vel, k_pos, k_vel, k_force):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, float, float, float, float, float, float, float, float, float, float, float) -> Tensor
    dof_pos_dif = dof_pos - dof_pos_gt

    dof_pos_sqdif_sum = torch.sum(dof_pos_dif.pow(2), dim=-1)

    dof_vel_dif = dof_vel - dof_vel_gt
    dof_vel_sqdif_sum = torch.sum(dof_vel_dif.pow(2), dim=-1)
    
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

    feet_vertical_force_dif = prev_feet_contact_forces[:, :, 2] - feet_contact_forces[:, :, 2]
    zero_comp = torch.zeros_like(feet_vertical_force_dif,
                                 dtype=feet_vertical_force_dif.dtype, device=feet_vertical_force_dif.device)
    feet_force_terms = torch.maximum(zero_comp, feet_vertical_force_dif)
    feet_force_terms_sum = torch.sum(feet_force_terms, dim=-1)

    r_dof_pos = w_dof_pos * torch.exp(-k_dof_pos * dof_pos_sqdif_sum)
    r_dof_vel = w_dof_vel * torch.exp(-k_dof_vel * dof_vel_sqdif_sum)
    r_pos = w_pos * torch.exp(-k_pos * rbj_pos_sqnorm_sum)
    r_vel = w_vel * torch.exp(-k_vel * rbj_vel_sqnorm_sum)
    #  according to the paper of QuestSim k_force should not be multiplied with -1
    #  but it does not make much sense to have better reward when difference is big
    #  (since we are trying to avoid that) + it can create inf values.
    #  Therefore try with negative and with clipping the term that goes inside of the exponential
    #  option 1
    r_force = w_force * torch.exp(-k_force * feet_force_terms_sum)
    # option 2 ?
    #  exp_term = k_force * feet_force_terms_sum
    #  exp_term[exp_term > 50.0] = 50.0
    #  r_force = w_force * torch.exp(exp_term)

    total_reward = r_dof_pos + r_dof_vel + r_pos + r_vel + r_force
    return total_reward

#questsim but angles of the DOFs as points in unit circle
@torch.jit.script
def compute_reward_v1(
        dof_pos, dof_pos_gt,
        dof_vel, dof_vel_gt,
        rigid_body_pos, rigid_body_pos_gt,
        rigid_body_vel, rigid_body_vel_gt,
        rigid_body_joints_indices,
        feet_contact_forces, prev_feet_contact_forces,
        termination_heights, key_bodies_ids, fall_penalty,
        w_dof_pos, w_dof_vel, w_pos, w_vel, w_force,
        k_dof_pos, k_dof_vel, k_pos, k_vel, k_force):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, float, float, float, float, float, float, float, float, float, float, float) -> Tensor

    #  angles in radians and angle velocities in radian per second
    #  transform into points in unit circle and estimate euclidean distance
    dof_pos_gt_cos = torch.cos(dof_pos_gt)
    dof_pos_gt_sin = torch.sin(dof_pos_gt)
    dof_pos_gt_ucp = torch.stack([dof_pos_gt_cos, dof_pos_gt_sin], dim=-1)
    dof_pos_cos = torch.cos(dof_pos)
    dof_pos_sin = torch.sin(dof_pos)
    dof_pos_ucp = torch.stack([dof_pos_cos, dof_pos_sin], dim=-1)

    dof_pos_ucp_dif = dof_pos_ucp - dof_pos_gt_ucp
    dof_pos_norm = torch.linalg.norm(dof_pos_ucp_dif, dim=-1)

    dof_pos_sqdif_sum = torch.sum(dof_pos_norm.pow(2), dim=-1)

    dof_vel_dif = dof_vel - dof_vel_gt
    dof_vel_sqdif_sum = torch.sum(dof_vel_dif.pow(2), dim=-1)

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

    feet_vertical_force_dif = prev_feet_contact_forces[:, :, 2] - feet_contact_forces[:, :, 2]
    zero_comp = torch.zeros_like(feet_vertical_force_dif,
                                 dtype=feet_vertical_force_dif.dtype, device=feet_vertical_force_dif.device)
    feet_force_terms = torch.maximum(zero_comp, feet_vertical_force_dif)
    feet_force_terms_sum = torch.sum(feet_force_terms, dim=-1)

    r_dof_pos = w_dof_pos * torch.exp(-k_dof_pos * dof_pos_sqdif_sum)
    r_dof_vel = w_dof_vel * torch.exp(-k_dof_vel * dof_vel_sqdif_sum)
    r_pos = w_pos * torch.exp(-k_pos * rbj_pos_sqnorm_sum)
    r_vel = w_vel * torch.exp(-k_vel * rbj_vel_sqnorm_sum)
    # according to the paper of QuestSim k_force should not be multiplied with -1
    # but it does not make much sense to have better reward when difference is big
    # (since we are trying to avoid that) + it can create inf values.
    # Therefore try with negative and with clipping the term that goes inside of the exponential

    r_force = w_force * torch.exp(-k_force * feet_force_terms_sum)

    total_reward = r_dof_pos + r_dof_vel + r_pos + r_vel + r_force
    return total_reward

# using an extra penalty when it falls
@torch.jit.script
def compute_reward_v2(
        dof_pos, dof_pos_gt,
        dof_vel, dof_vel_gt,
        rigid_body_pos, rigid_body_pos_gt,
        rigid_body_vel, rigid_body_vel_gt,
        rigid_body_joints_indices,
        feet_contact_forces, prev_feet_contact_forces,
        termination_heights, key_bodies_ids, fall_penalty,
        w_dof_pos, w_dof_vel, w_pos, w_vel, w_force,
        k_dof_pos, k_dof_vel, k_pos, k_vel, k_force):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, float, float, float, float, float, float, float, float, float, float, float) -> Tensor

    #  angles in radians and angle velocities in radian per second
    #  transform into points in unit circle and estimate euclidean distance
    dof_pos_gt_cos = torch.cos(dof_pos_gt)
    dof_pos_gt_sin = torch.sin(dof_pos_gt)
    dof_pos_gt_ucp = torch.stack([dof_pos_gt_cos, dof_pos_gt_sin], dim=-1)
    dof_pos_cos = torch.cos(dof_pos)
    dof_pos_sin = torch.sin(dof_pos)
    dof_pos_ucp = torch.stack([dof_pos_cos, dof_pos_sin], dim=-1)

    dof_pos_ucp_dif = dof_pos_ucp - dof_pos_gt_ucp
    dof_pos_norm = torch.linalg.norm(dof_pos_ucp_dif, dim=-1)

    dof_pos_sqdif_sum = torch.sum(dof_pos_norm.pow(2), dim=-1)

    dof_vel_dif = dof_vel - dof_vel_gt
    dof_vel_sqdif_sum = torch.sum(dof_vel_dif.pow(2), dim=-1)

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

    feet_vertical_force_dif = prev_feet_contact_forces[:, :, 2] - feet_contact_forces[:, :, 2]
    zero_comp = torch.zeros_like(feet_vertical_force_dif,
                                 dtype=feet_vertical_force_dif.dtype, device=feet_vertical_force_dif.device)
    feet_force_terms = torch.maximum(zero_comp, feet_vertical_force_dif)
    feet_force_terms_sum = torch.sum(feet_force_terms, dim=-1)

    r_dof_pos = w_dof_pos * torch.exp(-k_dof_pos * dof_pos_sqdif_sum)
    r_dof_vel = w_dof_vel * torch.exp(-k_dof_vel * dof_vel_sqdif_sum)
    r_pos = w_pos * torch.exp(-k_pos * rbj_pos_sqnorm_sum)
    r_vel = w_vel * torch.exp(-k_vel * rbj_vel_sqnorm_sum)
    # according to the paper of QuestSim k_force should not be multiplied with -1
    # but it does not make much sense to have better reward when difference is big
    # (since we are trying to avoid that) + it can create inf values.
    # Therefore try with negative and with clipping the term that goes inside of the exponential

    r_force = w_force * torch.exp(-k_force * feet_force_terms_sum)


    total_reward = r_dof_pos + r_dof_vel + r_pos + r_vel + r_force

    joints_heights = rigid_body_pos[:, :, 2]
    has_fallen = joints_heights < termination_heights
    # ignore the keyBodies (feet and hands)
    has_fallen[:, key_bodies_ids] = False
    #ignore if reference motion requires that the joints goes below this point
    joints_heights_gt = rigid_body_pos_gt[:, :, 2]
    has_fallen_gt = joints_heights_gt < termination_heights
    has_fallen[has_fallen_gt] = False
    # penalize env where it a joint has fallen
    has_fallen = torch.any(has_fallen, dim=-1)
    total_reward[has_fallen] = fall_penalty
    return total_reward