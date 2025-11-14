import numpy as np

class Preprocessor:
    
    # 定義任務數量為 3
    NUM_TASKS = 3  # 3 個任務

    def get_task_onehot(self, info, batch_size):
        """
        獲取任務索引，並返回 one-hot 向量。
        one-hot 向量是用來表示任務類別的一種方式，例如任務0會表示成 [1,0,0]。
        參數：
            info: 包含任務索引的資訊字典
            batch_size: 當前批次樣本數量
        回傳：
            onehot: 大小為 (batch_size, NUM_TASKS) 的 one-hot 向量
        """
        task_idx = 0  # 默認任務索引為0
        
        # 如果 info 裡面有 'task_index'，就提取它
        if 'task_index' in info:
            raw_task_idx = info['task_index']
            
            # 安全地處理不同格式的任務索引
            if isinstance(raw_task_idx, np.ndarray) and raw_task_idx.size > 0:
                task_idx = int(raw_task_idx.flatten()[0])  # 將 numpy array 展平後取第一個元素
            elif isinstance(raw_task_idx, (int, float, np.integer, np.floating)):
                task_idx = int(raw_task_idx)  # 如果是單一數字就直接轉成整數
        
        # 創建一個全0矩陣，大小為 (batch_size, NUM_TASKS)
        onehot = np.zeros((batch_size, self.NUM_TASKS), dtype=np.float32)
        if 0 <= task_idx < self.NUM_TASKS:
            onehot[:, task_idx] = 1.0  # 將對應任務位置設為1
        
        return onehot

    def quat_rotate_inverse(self, q: np.ndarray, v: np.ndarray):
        """
        使用四元數 q 的逆旋轉向量 v。
        四元數常用來表示三維空間中的旋轉。
        參數：
            q: 四元數，形狀為 (N,4)
            v: 需要旋轉的向量，形狀為 (N,3)
        回傳：
            被旋轉後的向量
        """
        # 如果 q 是一維的，將它擴展為批次形式 (1,4)
        if len(q.shape) == 1:
            q = np.expand_dims(q, axis=0)
            
        # 如果 v 是一維的，將它擴展為批次形式 (1,3)
        if v.shape == (3,):
            v = np.expand_dims(v, axis=0)

        q_w = q[:,[-1]]  # 四元數的純量部分 w
        q_vec = q[:,:3]  # 四元數的向量部分 x,y,z
        
        # 四元數旋轉公式拆解
        a = v * (2.0 * q_w**2 - 1.0)
        b = np.cross(q_vec, v) * (q_w * 2.0)
        c_scale = 2.0 * np.sum(q_vec * v, axis=1, keepdims=True)
        c = q_vec * c_scale
        
        return a - b + c  # 返回旋轉後的向量

    def modify_state(self, obs, info):
        """
        將原始觀測 obs 和額外 info 處理成神經網路可以用的形式。
        最後輸出的一維向量長度應為 87。
        """
        # ----------- 修正 1：保證 info 是 dict ----------- 
        # 有些情況 info 是 list 或 tuple，需要取第一個元素
        if isinstance(info, (list, tuple)) and len(info) > 0:
            info = info[0]

        # 保證 obs 是批次形式 (N, D)
        if len(obs.shape) == 1:
            obs = np.expand_dims(obs, axis=0)

        N = obs.shape[0]  # 批次大小
        task_onehot = self.get_task_onehot(info, N)  # 取得任務 one-hot 向量
        
        # 取出機器人的位置和速度資訊
        robot_qpos = obs[:, :12]  # 前12維是關節位置
        robot_qvel = obs[:, 12:24]  # 接下來12維是關節速度
        processed_features = [robot_qpos, robot_qvel]

        # 處理四元數資訊，用來計算重力方向
        quat_data = info.get("robot_quat")
        if quat_data is None or not isinstance(quat_data, np.ndarray) or quat_data.size == 0:
            project_gravity = np.zeros((N, 3), dtype=np.float32)
        else:
            # 將地球重力向量 [0,0,-1] 用四元數旋轉逆向旋轉到機器人座標系
            project_gravity = self.quat_rotate_inverse(quat_data, np.array([0.0, 0.0, -1.0]))
        processed_features.append(project_gravity)
        
        # 需要處理的其他資訊鍵
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
            target_D = 9 if key == "defender_xpos" else 3  # defender_xpos 特殊為9維
            data = info.get(key)

            # 如果資料缺失或格式不對，補零
            if data is None or not isinstance(data, np.ndarray) or data.size == 0:
                data_processed = np.zeros((N, target_D), dtype=np.float32)
            else:
                data_processed = np.array(data, dtype=np.float32)
                # 如果是一維向量，檢查長度是否符合
                if data_processed.ndim == 1:
                    if data_processed.shape[0] != target_D:
                        data_processed = np.zeros((N, target_D), dtype=np.float32)
                    else:
                        data_processed = np.tile(data_processed, (N, 1))  # 複製成批次形式
                elif data_processed.ndim == 2:
                    if data_processed.shape != (N, target_D):
                        data_processed = np.zeros((N, target_D), dtype=np.float32)
                else:
                    data_processed = np.zeros((N, target_D), dtype=np.float32)

            processed_features.append(data_processed)  # 加入特徵列表

        processed_features.append(task_onehot)  # 加入任務 one-hot
        processed_obs = np.hstack(processed_features)  # 水平堆疊成一個長向量

        # 確保最後的特徵長度是87
        if processed_obs.shape[1] != 87:
            print("DEBUG INFO:")
            for i, f in enumerate(processed_features):
                print(f"  Feature {i}: shape={f.shape}")
            raise ValueError(f"Feature stacking failed: Expected 87 dimensions, got {processed_obs.shape[1]} (N={N}).")

        return processed_obs.astype(np.float32)  # 最終返回32位浮點數
