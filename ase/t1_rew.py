import torch


@torch.jit.script
def compute_reward(
        dof_pos, dof_pos_gt,
        dof_vel, dof_vel_gt,
        rigid_body_pos, rigid_body_pos_gt,
        rigid_body_vel, rigid_body_vel_gt,
        rigid_body_joints_indices,
        feet_contact_forces, prev_feet_contact_forces,
        w_dof_pos=0.4, w_dof_vel=0.1, w_pos=0.2, w_vel=0.1, w_force=0.2,
        k_dof_pos=40.0, k_dof_vel=0.3, k_pos=6.0, k_vel=2.0, k_force=0.01):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, float, float, float, float, float, float, float, float, float, float) -> Tensor
    dof_pos_dif = dof_pos - dof_pos_gt
    dof_pos_sqdif_sum = torch.sum(dof_pos_dif*dof_pos_dif, dim=-1)

    dof_vel_dif = dof_vel - dof_vel_gt
    dof_vel_sqdif_sum = torch.sum(dof_vel_dif * dof_vel_dif, dim=-1)
    
    rbj_pos = rigid_body_pos[:, rigid_body_joints_indices, :]
    rbj_pos_gt = rigid_body_pos_gt[:, rigid_body_joints_indices, :]
    rbj_vel = rigid_body_vel[:, rigid_body_joints_indices, :]
    rbj_vel_gt = rigid_body_vel_gt[:, rigid_body_joints_indices, :]

    rbj_pos_dif = rbj_pos - rbj_pos_gt
    rbj_pos_norm = torch.linalg.norm(rbj_pos_dif, dim=-1)
    rbj_pos_sqnorm_sum = torch.sum(rbj_pos_norm*rbj_pos_norm, dim=-1)

    rbj_vel_dif = rbj_vel - rbj_vel_gt
    rbj_vel_norm = torch.linalg.norm(rbj_vel_dif, dim=-1)
    rbj_vel_sqnorm_sum = torch.sum(rbj_vel_norm * rbj_vel_norm, dim=-1)

    feet_vertical_force_dif = prev_feet_contact_forces[:, :, 2] - feet_contact_forces[:, :, 2]
    zero_comp = torch.zeros_like(feet_vertical_force_dif,
                                 dtype=feet_vertical_force_dif.dtype, device=feet_vertical_force_dif.device)
    feet_force_terms = torch.maximum(zero_comp, feet_vertical_force_dif)
    feet_force_terms_sum = torch.sum(feet_force_terms, dim=-1)

    total_reward = w_dof_pos * torch.exp(-k_dof_pos * dof_pos_sqdif_sum) \
                   + w_dof_vel * torch.exp(-k_dof_vel * dof_vel_sqdif_sum) \
                   + w_pos * torch.exp(-k_pos * rbj_pos_sqnorm_sum) \
                   + w_vel * torch.exp(-k_vel * rbj_vel_sqnorm_sum) \
                   + w_force * torch.exp(k_force * feet_force_terms_sum)  # is it correct that the last one is not negative?
    return total_reward
