

from isaacgym.torch_utils import quat_rotate, quat_mul
from utils import torch_utils
import torch


# adjusted from quaternion_to_matrix of
# https://github.com/facebookresearch/pytorch3d/blob/main/pytorch3d/transforms/rotation_conversions.py
@torch.jit.script
def quat_to_matrix_obs_repr(flat_rot, original_shape):
    # type: (Tensor, List[int]) -> Tensor
    quaternions = flat_rot.view(original_shape[0] * original_shape[1], original_shape[2])
    r, i, j, k = torch.unbind(quaternions, -1)

    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    o = o.view(quaternions.shape[:-1] + (3, 3))
    o = o[:, :, :2]  # just taking the 2 columns of the matrix
    o = o.reshape(original_shape[0], original_shape[1] * 3 * 2)
    return o


@torch.jit.script
def _estimate_sframe(body_pos, body_rot):
    # type: (Tensor, Tensor) -> Tuple[Tensor, Tensor]
    root_pos = body_pos[:, 0, :]
    root_rot = body_rot[:, 0, :]

    sframe_pos = torch.clone(root_pos)
    sframe_pos[:, 2] = 0  # setting the height to the floor; just works since floor is always 0
    sframe_heading_rot = torch_utils.calc_heading_quat_inv(root_rot)
    return sframe_pos, sframe_heading_rot


@torch.jit.script
def estimate_srel_measures(body_pos, body_rot, body_vel, body_ang_vel, sframe_pos, sframe_heading_rot):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]

    sframe_heading_rot = sframe_heading_rot.unsqueeze(-2)
    sframe_heading_rot = sframe_heading_rot.repeat((1, body_pos.shape[1], 1))
    flat_sframe_heading_rot = sframe_heading_rot.reshape(sframe_heading_rot.shape[0] * sframe_heading_rot.shape[1],
                                                         sframe_heading_rot.shape[2])
    sframe_pos = sframe_pos.unsqueeze(-2)

    # transform pos
    tr_pos = body_pos - sframe_pos
    flat_srel_pos = tr_pos.view(body_pos.shape[0] * body_pos.shape[1], body_pos.shape[2])
    flat_srel_pos = quat_rotate(flat_sframe_heading_rot, flat_srel_pos)
    flat_srel_pos = flat_srel_pos.view(body_pos.shape[0], body_pos.shape[1] * body_pos.shape[2])

    # transform rot
    flat_srel_rot = body_rot.view(body_rot.shape[0] * body_rot.shape[1], body_rot.shape[2])
    flat_srel_rot = quat_mul(flat_sframe_heading_rot, flat_srel_rot)
    flat_srel_rot = flat_srel_rot.view(body_rot.shape[0], body_rot.shape[1] * body_rot.shape[2])

    if body_vel.shape[0] == 0 and body_ang_vel.shape[0] == 0:  # a dummy tensor was used to not send data
        dummy_tensor = torch.empty(0, device=flat_srel_pos.device)
        return flat_srel_pos, flat_srel_rot, dummy_tensor, dummy_tensor
    else:
        # transform vel
        flat_srel_vel = body_vel.view(body_vel.shape[0] * body_vel.shape[1], body_vel.shape[2])
        flat_srel_vel = quat_rotate(flat_sframe_heading_rot, flat_srel_vel)
        flat_srel_vel = flat_srel_vel.view(body_vel.shape[0], body_vel.shape[1] * body_vel.shape[2])

        # transform angular velocity
        flat_srel_ang_vel = body_ang_vel.view(body_ang_vel.shape[0] * body_ang_vel.shape[1], body_ang_vel.shape[2])
        flat_srel_ang_vel = quat_rotate(flat_sframe_heading_rot, flat_srel_ang_vel)
        flat_srel_ang_vel = flat_srel_ang_vel.view(body_ang_vel.shape[0], body_ang_vel.shape[1] * body_ang_vel.shape[2])

        return flat_srel_pos, flat_srel_rot, flat_srel_vel, flat_srel_ang_vel


@torch.jit.script
def get_obs_user(track_poses_acc, track_rots_acc, sframe_pos, sframe_heading_rot):
    # type: (Tensor, Tensor, Tensor, Tensor) -> Tensor
    dummy_tensor = torch.empty(0, device=track_poses_acc.device)
    flat_srel_poses, flat_srel_rots, _, _ = estimate_srel_measures(track_poses_acc, track_rots_acc,
                                                                   dummy_tensor, dummy_tensor,
                                                                   sframe_pos, sframe_heading_rot)

    flat_srel_rots = quat_to_matrix_obs_repr(flat_srel_rots, list(track_rots_acc.shape))
    obs = torch.cat((flat_srel_poses, flat_srel_rots), dim=-1)
    return obs


@torch.jit.script
def get_obs_sim(_rigid_body_pos, _rigid_body_rot, _rigid_body_vel, _rigid_body_ang_vel, _rigid_body_joints_indices,
                dof_pos, dof_vel, feet_contact_forces, sframe_pos, sframe_heading_rot):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor) -> Tensor
    flat_dof_pos = dof_pos
    flat_dof_vel = dof_vel

    jrb_pos = _rigid_body_pos[:, _rigid_body_joints_indices, :]
    jrb_rot = _rigid_body_rot[:, _rigid_body_joints_indices, :]
    jrb_vel = _rigid_body_vel[:, _rigid_body_joints_indices, :]
    jrb_ang_vel = _rigid_body_ang_vel[:, _rigid_body_joints_indices, :]
    flat_srel_pos, flat_srel_rot, flat_srel_vel, flat_srel_ang_vel = estimate_srel_measures(jrb_pos,
                                                                                            jrb_rot,
                                                                                            jrb_vel,
                                                                                            jrb_ang_vel,
                                                                                            sframe_pos,
                                                                                            sframe_heading_rot)
    flat_srel_rot = quat_to_matrix_obs_repr(flat_srel_rot, list(jrb_rot.shape))

    flat_feet_contact_forces = feet_contact_forces.reshape(feet_contact_forces.shape[0],
                                                           feet_contact_forces.shape[1] * feet_contact_forces.shape[2])

    obs = torch.cat(
        (flat_dof_pos, flat_dof_vel, flat_srel_pos, flat_srel_rot, flat_srel_vel,
         flat_srel_ang_vel, flat_feet_contact_forces),
        dim=-1
    )
    return obs

@torch.jit.script
def get_obs(_rigid_body_pos, _rigid_body_rot, _rigid_body_vel, _rigid_body_ang_vel, _rigid_body_joints_indices,
            dof_pos, dof_vel, feet_contact_forces, track_poses_acc, track_rots_acc, track_headset_index,
            num_track_dev,
            imit_motion_height, humanoid_height, include_global):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, bool) -> Tensor

    sframe_pos, sframe_heading_rot = _estimate_sframe(_rigid_body_pos, _rigid_body_rot)
    obs_sim = get_obs_sim(_rigid_body_pos, _rigid_body_rot, _rigid_body_vel, _rigid_body_ang_vel,
                          _rigid_body_joints_indices,
                          dof_pos, dof_vel, feet_contact_forces, sframe_pos, sframe_heading_rot)



    obs_user = get_obs_user(track_poses_acc, track_rots_acc, sframe_pos, sframe_heading_rot)
    #  additionally includes the global position of the root
    root_global_pos = _rigid_body_pos[:, 0, :].reshape((obs_sim.shape[0], -1))
    #  and first track device
    first_step_obs = track_poses_acc[:, :num_track_dev, :]
    headset_global_pos = first_step_obs[:, track_headset_index, :].reshape((obs_sim.shape[0], -1))

    obs_userheight = torch.reshape(imit_motion_height, shape=(imit_motion_height.shape[0], 1))
    obs_simheight = torch.reshape(humanoid_height, shape=(humanoid_height.shape[0], 1))

    if include_global:
        o = torch.cat((root_global_pos, headset_global_pos, obs_sim, obs_user, obs_userheight, obs_simheight), dim=-1)
    else:
        o = torch.cat((obs_sim, obs_user, obs_userheight, obs_simheight), dim=-1)
    return o
