import pickle
import argparse
import numpy as np
import os
import shutil


def phrase_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-Mb-file', type=str, required=True, default='')
    parser.add_argument('--input-SMPL-dir', type=str, required=True, default='/nfs/turbo/coe-shdpm/leyang/VEHS-7M/Mesh/SMPL_obj_pkl/')
    parser.add_argument('--input-SMPL-store-dir', type=str, required=True, default='/nfs/turbo/coe-shdpm/leyang/VEHS-7M/Mesh/SMPL_pkl/')
    parser.add_argument('--input-3DposeMB-file', type=str, required=True, default='/nfs/turbo/coe-shdpm/leyang/VEHS_MB/RTM2D_VEHS_config6/VEHS_6D_downsample5_keep1_config6_modified_RTM2D.pkl')
    # todo: change to gt 2d with 66 keypoints for training
    arg = parser.parse_args()
    return arg


if __name__ == '__main__':
    arg = phrase_arg()

    # Step 1: copy smpl_merge results to a seperated folder
    # from args.dir -> S01 -> Activity00_stageii -> merged.pkl
    # to args.store_dir -> S01 -> Activity00.pkl
    merged_files = []
    for root, dirs, files in os.walk(arg.input_SMPL_dir):
        for file in files:
            if file == "merged.pkl":
                print(f"Processing {root}")
                subject_name = root.split('/')[-2]
                activity_name = root.split('/')[-1].split('_')[0]
                output_path = os.path.join(arg.input_SMPL_store_dir, subject_name, activity_name + ".pkl")
                # copy file
                if not os.path.exists(os.path.dirname(output_path)):
                    os.makedirs(os.path.dirname(output_path))
                shutil.copy(os.path.join(root, "merged.pkl"), output_path)
                merged_files.append(output_path)

    # Step 2: read 3DposeMB file
    with open(arg.input_3DposeMB_file, "rb") as f:
        data_old = pickle.load(f)

    # Step 3: project SMPL to camera view

    # Step 4: merge SMPL and 3DposeMB
