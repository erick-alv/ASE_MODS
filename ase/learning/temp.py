class Sample:

    def log_error(self, motion_name):
        joints_indices = self.env_config["env"]["asset"]["jointsIndices"]
        joints_names = self.env_config["env"]["asset"]["jointsNames"]
        accumulated_pos_error = self.accumulated_pos_error
        accumulated_rot_error = self.accumulated_rot_error

        for step in range(accumulated_pos_error.shape[0]):
            for n in range(accumulated_pos_error.shape[1]):
                for i in range(len(joints_indices)):
                    j_id = joints_indices[i]
                    j_name = joints_names[i]
                    self.writer.add_scalar(
                        f'{motion_name}/env_{n}/{j_name}/pos_error',
                        accumulated_pos_error[step, n, j_id], step
                    )
                    self.writer.add_scalar(
                        f'{motion_name}/env_{n}/{j_name}/rot_error',
                        accumulated_rot_error[step, n, j_id], step
                    )

        mean_pos_error_per_joint = torch.mean(accumulated_pos_error, dim=0)
        mean_rot_error_per_joint = torch.mean(accumulated_rot_error, dim=0)
        for n in range(mean_pos_error_per_joint.shape[0]):
            for i in range(len(joints_indices)):
                j_id = joints_indices[i]
                j_name = joints_names[i]
                self.writer.add_scalar(
                    f'{motion_name}/env_{n}/{j_name}/as_pos_error',
                    mean_pos_error_per_joint[n, j_id], 0
                )
                self.writer.add_scalar(
                    f'{motion_name}/env_{n}/{j_name}/as_rot_error',
                    mean_rot_error_per_joint[n, j_id], 0
                )

        mean_pos_error_per_joint = mean_pos_error_per_joint[:, joints_indices]  # selects the joints
        mean_rot_error_per_joint = mean_rot_error_per_joint[:, joints_indices]
        mean_pos_error = torch.mean(mean_pos_error_per_joint, dim=1)
        mean_rot_error = torch.mean(mean_rot_error_per_joint, dim=1)
        for n in range(mean_pos_error.shape[0]):
            self.writer.add_scalar(
                f'{motion_name}/env_{n}/as_aj_pos_error',
                mean_pos_error[n], 0
            )
            self.writer.add_scalar(
                f'{motion_name}/env_{n}/as_aj_rot_error',
                mean_rot_error[n], 0
            )