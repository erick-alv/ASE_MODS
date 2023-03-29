import ast
from isaacgym.torch_utils import quat_mul, quat_rotate

from poselib.poselib import quat_from_angle_axis
import torch


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

def controllers_to_training_orientation(input_el):
    """
    This transformation is not done to transform the input coordinate system to the coordinate system of the simulation.
    The contollers in the systhetic data had another default rotation and this transformation makes it seems
    that the input pose of controllers have the same default orientation.
    (Just controllers; not HMD)
    """
    pos_tensor, rot_tensor, buttons_tensor = input_el




    new_rot_tensor = []
    for i in range(rot_tensor.size()[0]):
        if i == 0:#the first one is the HMD; this one is not changed
            new_rot_tensor.append(rot_tensor[i])
        else:

            # -90 degrees around the y-axis in the coordinates system of isaac gym
            rot_quat = to_torch([0., -0.7071068, 0., 0.7071068], device=rot_tensor.device)
            new_rot = quat_mul(rot_tensor[i], rot_quat)

            new_rot_tensor.append(new_rot)
    new_rot_tensor = torch.stack(new_rot_tensor)

    return pos_tensor, new_rot_tensor, buttons_tensor


def all_transforms(input_info):
    return controllers_to_training_orientation(transform_coord_system(parse_pos_and_rot(to_torch(tr_str(input_info)))))

def check_if_button_A_pressed(transformed_element):
    #get button input
    buttons_tensor = transformed_element[2]
    return buttons_tensor[0] == 1

# if __name__ == '__main__':
#     rot_tensor = to_torch([0.0, 0.0, -0.3827, 0.9239], device='cuda:0')
#     y_axis = to_torch([0.0, 1.0, 0.0], device=rot_tensor.device)
#     y_axis = quat_rotate(torch.unsqueeze(rot_tensor, 0), torch.unsqueeze(y_axis, 0))
#     y_axis = y_axis[0]
#     print(y_axis)
