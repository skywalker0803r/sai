# submit_adapter.py
import os
import subprocess
import json
from pathlib import Path

# 假設 DreamerV3 checkpoints 存在 logdir/checkpoints/
CHECKPOINT_DIR = "/path/to/dreamer/logdir/checkpoints"
MODEL_ARCHIVE = "/path/to/model_for_submission.tar.gz"
COMP_ID = "booster-soccer-showdown"
API_KEY = "sai_..."  # 你的 SAI api key

def package_model(checkpoint_dir, output_path):
    # 簡單 tar 打包所有 checkpoint（或挑選你要的 file）
    p = Path(output_path)
    if p.exists():
        p.unlink()
    subprocess.run(["tar", "-czf", output_path, "-C", checkpoint_dir, "."], check=True)
    print("Packaged model:", output_path)

def submit_model(package_path, comp_id, api_key):
    # 假設你把官方 submit_sai.py 放在 local 仓库
    submit_script = "submit_sai.py"
    cmd = ["python", submit_script, "--comp_id", comp_id, "--api_key", api_key, "--model", package_path]
    print("Running submit command:", " ".join(cmd))
    subprocess.run(cmd, check=True)

if __name__ == "__main__":
    package_model(CHECKPOINT_DIR, MODEL_ARCHIVE)
    submit_model(MODEL_ARCHIVE, COMP_ID, API_KEY)
