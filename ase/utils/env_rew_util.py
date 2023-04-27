import torch

#todo
#@torch.jit.script
def compute_reward(
        dof_pos, dof_pos_gt,
        dof_vel, dof_vel_gt,
        rigid_body_pos, rigid_body_pos_gt,
        rigid_body_vel, rigid_body_vel_gt,
        rigid_body_joints_indices,
        feet_contact_forces, prev_feet_contact_forces,
        w_dof_pos, w_dof_vel, w_pos, w_vel, w_force,
        k_dof_pos, k_dof_vel, k_pos, k_vel, k_force):
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



    r_dof_pos = w_dof_pos * torch.exp(-k_dof_pos * dof_pos_sqdif_sum)
    r_dof_vel = w_dof_vel * torch.exp(-k_dof_vel * dof_vel_sqdif_sum)
    r_pos = w_pos * torch.exp(-k_pos * rbj_pos_sqnorm_sum)
    r_vel = w_vel * torch.exp(-k_vel * rbj_vel_sqnorm_sum)
    #todo according to the paper of QuestSim k_force should not be multiplied with -1
    # but it does not make much sense to have better reward when difference is big
    # (since we are trying to avoid that) + it can create inf values.
    # Therefore try with negative and with clipping the term that goes inside of the exponential

    #option 1
    r_force = w_force * torch.exp(-k_force * feet_force_terms_sum)
    # option 2
    #exp_term = k_force * feet_force_terms_sum
    #exp_term[exp_term > 50.0] = 50.0
    #r_force = w_force * torch.exp(exp_term)


    #TODO (this check must be done in env, not here) according the explanation given in paper this implies that when the foot is
    # touching the ground the force in up-down axis should be pointing upwards. In this
    # way it penalizes correctly abrupt "decrease" (not unload the fit and then lift)


    total_reward =  r_dof_pos + r_dof_vel + r_pos + r_vel + r_force
    return total_reward
