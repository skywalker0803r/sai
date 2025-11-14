# preprocessor.py
import numpy as np

class Preprocessor():
    def get_task_onehot(self, info):
        if 'task_index' in info:
            t = info['task_index']
            # convert to one-hot of length 3
            onehot = np.zeros(3, dtype=np.float32)
            if isinstance(t, (list, np.ndarray)):
                # if already one-hot or index array
                if len(t) == 3:
                    return np.array(t, dtype=np.float32)
                t = int(t[0])
            onehot[int(t)] = 1.0
            return onehot
        else:
            return np.zeros(3, dtype=np.float32)

    def quat_rotate_inverse(self, q: np.ndarray, v: np.ndarray):
        # q expected shape (batch, 4) last component w, vector part first 3
        if q.ndim == 1:
            q = np.expand_dims(q, axis=0)
        q_w = q[:, -1:]
        q_vec = q[:, :3]
        # broadcasting
        a = v * (2.0 * q_w**2 - 1.0)
        b = np.cross(q_vec, v) * (q_w * 2.0)
        # dot product for each batch
        dot = (q_vec * v).sum(axis=1, keepdims=True)
        c = q_vec * (dot * 2.0)
        return a - b + c

    def modify_state(self, obs, info):
        # obs -> ensure batch dim
        if len(obs.shape) == 1:
            obs = np.expand_dims(obs, axis=0)

        # ensure all info arrays have batch dim
        for k, v in info.items():
            if isinstance(v, np.ndarray) and v.ndim == 1:
                info[k] = np.expand_dims(v, axis=0)

        # robot qpos/qvel
        robot_qpos = obs[:, :12]
        robot_qvel = obs[:, 12:24]
        quat = info.get("robot_quat", np.zeros((obs.shape[0], 4), dtype=np.float32))
        base_ang_vel = info.get("robot_gyro", np.zeros((obs.shape[0], 3), dtype=np.float32))
        project_gravity = self.quat_rotate_inverse(quat, np.array([0.0, 0.0, -1.0], dtype=np.float32))
        # gather features similar to official
        parts = [
            robot_qpos, robot_qvel, project_gravity, base_ang_vel,
            info.get("robot_accelerometer", np.zeros((obs.shape[0], 3), dtype=np.float32)),
            info.get("robot_velocimeter", np.zeros((obs.shape[0], 3), dtype=np.float32)),
            info.get("goal_team_0_rel_robot", np.zeros((obs.shape[0], 3), dtype=np.float32)),
            info.get("goal_team_1_rel_robot", np.zeros((obs.shape[0], 3), dtype=np.float32)),
            info.get("goal_team_0_rel_ball", np.zeros((obs.shape[0], 3), dtype=np.float32)),
            info.get("goal_team_1_rel_ball", np.zeros((obs.shape[0], 3), dtype=np.float32)),
            info.get("ball_xpos_rel_robot", np.zeros((obs.shape[0], 3), dtype=np.float32)),
            info.get("ball_velp_rel_robot", np.zeros((obs.shape[0], 3), dtype=np.float32)),
            info.get("ball_velr_rel_robot", np.zeros((obs.shape[0], 3), dtype=np.float32)),
            info.get("player_team", np.zeros((obs.shape[0], 2), dtype=np.float32)),
            info.get("goalkeeper_team_0_xpos_rel_robot", np.zeros((obs.shape[0], 3), dtype=np.float32)),
            info.get("goalkeeper_team_0_velp_rel_robot", np.zeros((obs.shape[0], 3), dtype=np.float32)),
            info.get("goalkeeper_team_1_xpos_rel_robot", np.zeros((obs.shape[0], 3), dtype=np.float32)),
            info.get("goalkeeper_team_1_velp_rel_robot", np.zeros((obs.shape[0], 3), dtype=np.float32)),
            info.get("target_xpos_rel_robot", np.zeros((obs.shape[0], 3), dtype=np.float32)),
            info.get("target_velp_rel_robot", np.zeros((obs.shape[0], 3), dtype=np.float32)),
            info.get("defender_xpos", np.zeros((obs.shape[0], 9), dtype=np.float32)),
        ]
        flat = np.hstack(parts)
        task_onehot = self.get_task_onehot(info)
        if task_onehot.ndim == 1:
            task_onehot = np.expand_dims(task_onehot, axis=0)
        out = np.hstack((flat, task_onehot))
        return out  # shape (1, N)
