import numpy as np

class Preprocessor:
    
    NUM_TASKS = 3 # 3 個任務

    def get_task_onehot(self, info, batch_size):
        """獲取 task_index 並返回 (N, 3) 的 one-hot 向量。"""
        task_idx = 0 
        
        if 'task_index' in info:
            raw_task_idx = info['task_index']
            
            # 安全地提取任務索引
            if isinstance(raw_task_idx, np.ndarray) and raw_task_idx.size > 0:
                task_idx = int(raw_task_idx.flatten()[0])
            elif isinstance(raw_task_idx, (int, float, np.integer, np.floating)):
                task_idx = int(raw_task_idx)
        
        onehot = np.zeros((batch_size, self.NUM_TASKS), dtype=np.float32)
        if 0 <= task_idx < self.NUM_TASKS:
            onehot[:, task_idx] = 1.0
        
        return onehot

    def quat_rotate_inverse(self, q: np.ndarray, v: np.ndarray):
        """
        將向量 v 旋轉四元數 q 的逆。已修正 N=1 時的索引錯誤。
        """
        # 確保 q 始終是 (N, 4) 的批次形式
        if len(q.shape) == 1:
            q = np.expand_dims(q, axis=0)
            
        # 確保 v 始終是 (N, 3) 的批次形式
        if v.shape == (3,):
            v = np.expand_dims(v, axis=0)

        q_w = q[:,[-1]] 
        q_vec = q[:,:3]
        
        a = v * (2.0 * q_w**2 - 1.0)
        b = np.cross(q_vec, v) * (q_w * 2.0)
        
        c_scale = 2.0 * np.sum(q_vec * v, axis=1, keepdims=True)
        c = q_vec * c_scale
        
        return a - b + c 

    def modify_state(self, obs, info):
      # ----------- 修正 1：保證 info 是 dict -----------
      if isinstance(info, (list, tuple)) and len(info) > 0:
          info = info[0]

      if len(obs.shape) == 1:
          obs = np.expand_dims(obs, axis=0)

      N = obs.shape[0] 
      task_onehot = self.get_task_onehot(info, N)
      
      robot_qpos = obs[:, :12] 
      robot_qvel = obs[:, 12:24] 
      processed_features = [robot_qpos, robot_qvel]

      quat_data = info.get("robot_quat")
      if quat_data is None or not isinstance(quat_data, np.ndarray) or quat_data.size == 0:
          project_gravity = np.zeros((N, 3), dtype=np.float32)
      else:
          project_gravity = self.quat_rotate_inverse(quat_data, np.array([0.0, 0.0, -1.0]))
      processed_features.append(project_gravity)
      
      info_keys = [
          "robot_gyro", "robot_accelerometer", "robot_velocimeter", 
          "goal_team_0_rel_robot", "goal_team_1_rel_robot", "goal_team_0_rel_ball",
          "goal_team_1_rel_ball", "ball_xpos_rel_robot", "ball_velp_rel_robot",
          "ball_velr_rel_robot", 
          "goalkeeper_team_0_xpos_rel_robot", "goalkeeper_team_0_velp_rel_robot",
          "goalkeeper_team_1_xpos_rel_robot", "goalkeeper_team_1_velp_rel_robot",
          "target_xpos_rel_robot", "target_velp_rel_robot", 
          "defender_xpos"
      ]

      for key in info_keys:
          target_D = 9 if key == "defender_xpos" else 3
          data = info.get(key)

          if data is None or not isinstance(data, np.ndarray) or data.size == 0:
              data_processed = np.zeros((N, target_D), dtype=np.float32)
          else:
              data_processed = np.array(data, dtype=np.float32)
              if data_processed.ndim == 1:
                  if data_processed.shape[0] != target_D:
                      data_processed = np.zeros((N, target_D), dtype=np.float32)
                  else:
                      data_processed = np.tile(data_processed, (N, 1))
              elif data_processed.ndim == 2:
                  if data_processed.shape != (N, target_D):
                      data_processed = np.zeros((N, target_D), dtype=np.float32)
              else:
                  data_processed = np.zeros((N, target_D), dtype=np.float32)

          processed_features.append(data_processed)

      processed_features.append(task_onehot)
      processed_obs = np.hstack(processed_features)

      if processed_obs.shape[1] != 87:
          print("DEBUG INFO:")
          for i, f in enumerate(processed_features):
              print(f"  Feature {i}: shape={f.shape}")
          raise ValueError(f"Feature stacking failed: Expected 87 dimensions, got {processed_obs.shape[1]} (N={N}).")

      return processed_obs.astype(np.float32)
