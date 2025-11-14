# submit_sai.py
"""
Universal SAI submission script
Adapted for DreamerV3 checkpoints
"""

import argparse
import os
from sai_rl import SAIClient

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--comp_id", type=str, required=True,
                        help="Competition ID, e.g. booster-soccer-showdown")
    parser.add_argument("--api_key", type=str, required=True,
                        help="Your SAI API key starting with sai_...")
    parser.add_argument("--model", type=str, required=True,
                        help="Path to model .tar.gz produced by submit_adapter.py")
    parser.add_argument("--entry", type=str, default="dreamerv3",
                        help="A name for your submission (optional)")
    args = parser.parse_args()

    if not os.path.exists(args.model):
        raise FileNotFoundError(f"Model file not found: {args.model}")

    print("🔗 Connecting to SAI...")
    sai = SAIClient(comp_id=args.comp_id, api_key=args.api_key)

    print("📤 Uploading model to SAI...")
    model_id = sai.upload_model(args.model, name=args.entry)
    print(f"Model uploaded. model_id = {model_id}")

    print("🏆 Submitting to competition leaderboard...")
    sub_id = sai.submit_model(model_id)
    print(f"Submission completed! submission_id = {sub_id}")

    print("\n🎉 Done! Check your leaderboard status on the website!")

if __name__ == "__main__":
    main()
