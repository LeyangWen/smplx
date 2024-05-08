# merges the output of the main transfer_model script

import torch
from pathlib import Path
import pickle
from scipy.spatial.transform import Rotation as R
import wandb
import time
import os
import shutil

KEYS = [
"transl",
"global_orient",
"body_pose",
"betas",
"left_hand_pose",
"right_hand_pose",
"jaw_pose",
"leye_pose",
"reye_pose",
"expression",
"vertices",
"joints",
"full_pose",
"v_shaped",
"faces"
]  # SMPLX keys

KEYS = ['transl',
        'global_orient',
        'body_pose',
        'betas',
        'vertices',
        'joints',
        'full_pose',
        'v_shaped',
        'faces'
         ]  # SMPL keys


IGNORED_KEYS = [
'transl',  # bug in transfer model set this to none, temporary fix
"vertices",
"faces",
"v_shaped"
]

def aggregate_rotmats(x):
    x = torch.cat(x, dim=0).detach().cpu().numpy()
    s = x.shape[:-2]
    x = R.from_matrix(x.reshape(-1, 3, 3)).as_rotvec()
    x = x.reshape(s[0], -1)
    return x

aggregate_function = {k: lambda x: torch.cat(x, 0).detach().cpu().numpy() for k in KEYS}
aggregate_function["betas"] = lambda x: torch.cat(x, 0).mean(0).detach().cpu().numpy()

for k in ["global_orient", "body_pose", "left_hand_pose", "right_hand_pose", "jaw_pose", "full_pose"]:
    aggregate_function[k] = aggregate_rotmats

def merge(output_dir, gender, store_dir=None):
    output_dir = Path(output_dir)
    assert output_dir.exists()
    assert output_dir.is_dir()

    # get list of all pkl files in output_dir with fixed length numeral names
    pkl_files = [f for f in output_dir.glob("*.pkl") if f.stem != "merged"]
    pkl_files = [f for f in sorted(pkl_files, key=lambda x: int(x.stem))]
    assert "merged.pkl" not in [f.name for f in pkl_files]

    merged = {}
    # iterate over keys and put all values in lists
    keys = set(KEYS) - set(IGNORED_KEYS)
    for k in keys:
        merged[k] = []
    start_time = time.time()
    for pkl_i, pkl_file in enumerate(pkl_files):
        wandb.log({"time": time.time() - start_time})
        if pkl_i % 128 != 0:  # Wen: temp fix for old bug, no need for new pkl files
            continue
        with open(pkl_file, "rb") as f:
            data = pickle.load(f)
        for k in keys:
            if k in data:
                merged[k].append(data[k])
    b = torch.cat(merged["betas"], 0)
    print("betas:")
    for mu, sigma in zip(b.mean(0), b.std(0)):
        print("  {:.3f} +/- {:.3f}".format(mu, sigma))

    # aggregate all values
    for k in keys:
        merged[k] = aggregate_function[k](merged[k])

    # add gender
    merged["gender"] = gender
    store_dir = f"{output_dir}/merged.pkl" if store_dir is None else store_dir
    # save merged data to same output_dir
    with open(store_dir, "wb") as f:
        pickle.dump(merged, f)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Merge output of transfer_model script')
    parser.add_argument('output_dir', type=str, help='output directory of transfer_model script')
    parser.add_argument('--gender', type=str, choices=['male', 'female', 'neutral'], help='gender of actor in motion sequence')
    parser.add_argument('--wandb-project', default='smpl-smplx', help='wandb project name')
    parser.add_argument('--wandb-name', default='test-merge', help='wandb run name')
    parser.add_argument(
        "--batch-moshpp",
        action="store_true",
        help="Batch process moshpp output, will use args.motion-file as a directory",
    )
    parser.add_argument('--SMPL-batch-store-dir', type=str, default='/nfs/turbo/coe-shdpm/leyang/VEHS-7M/Mesh/SMPL_pkl/')
    parser.add_argument("--batch-id", type=int)
    args = parser.parse_args()
    wandb.init(project=args.wandb_project, name=args.wandb_name, config=args)  # Initialize a new run
    if args.batch_moshpp:
        subject_name = f"S{args.batch_id:02d}"
        dir_name = os.path.join(args.output_dir, subject_name)
        male_subject_list = ['S01', 'S02', 'S04', 'S07', 'S08']
        female_subject_list = ['S03', 'S05', 'S06', 'S09', 'S10']
        if subject_name in male_subject_list:
            gender = "male"
        elif subject_name in female_subject_list:
            gender = "female"
        else:
            gender = "neutral"
        print(f"Overwriting args.gender for subject {args.batch_id} to {gender}")

        for activity in os.listdir(dir_name):
            merge_dir = os.path.join(dir_name, activity)
            if not os.path.isdir(merge_dir):
                continue
            print("@"*60)
            print(f"Processing {args.batch_id} - {activity}")
            activity_name = activity.split('_')[0]
            output_path = os.path.join(args.SMPL_batch_store_dir, subject_name, activity_name + ".pkl")
            if not os.path.exists(os.path.dirname(output_path)):
                os.makedirs(os.path.dirname(output_path))
            merge(merge_dir, gender, store_dir=output_path)
            print(f"Finished processing S{args.batch_id} - {activity}")
            print(f"Stored in {output_path}")

    else:
        merge(args.output_dir, args.gender)
    wandb.finish()
