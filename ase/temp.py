import torch
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
    # swap axes
    tr_m1 = to_torch([
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0]
    ])

    # change dir of second axis (since isaac gym uses positive y = left)
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


if __name__ == "__main__":
    in_str = '[ 0, 1.72, 0, 0, 0, 0, 1, -0.75, 1.42, 0, 0, 0, 0, 1, 0.75, 1.42, 0, 0, 0, 0, 1, 0, 0 ]\n'
    out_el = all_transforms(in_str)
    print(out_el)