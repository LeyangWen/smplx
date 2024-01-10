import argparse
import subprocess
import os
import time
import numpy as np
import torch
from smplx.body_models import SMPL
from matplotlib import pyplot as plt

from SMPLPose import SMPLPose
from SSPPOutput import SSPPV7Output


def wait_for_file_update(output_file, initial_mtime):
    current_mtime = os.stat(output_file).st_mtime
    while True:
        time.sleep(1)  # Wait for a short period of time before checking again
        if current_mtime != initial_mtime:  # The output file has been updated
            for _ in range(20):  # wait till the output file have finished writing
                if current_mtime != os.stat(output_file).st_mtime:
                    print(f"Waiting for the output file finish writing: {current_mtime} -> {os.stat(output_file).st_mtime}", end='\r')
                    current_mtime = os.stat(output_file).st_mtime
                    time.sleep(4)
                else:
                    print("Finished writing the output file", end="\r")
                    time.sleep(2)
                    break
            break


def parse_args():
    parser = argparse.ArgumentParser(description='SMPLPose')
    parser.add_argument('--small_sample', type=int, default=12, help='only process the first n files for testing, set to a large number to process all')
    parser.add_argument('--motion_smpl_base_dir', type=str, default=r'experiment\text2pose-20231113T194712Z-001\text2pose'
                        , help='motion smpl directory')
    parser.add_argument('--search_string', type=str, default='smpl_pose_72', help='search string for motion smpl files')

    ### text prompts ###
    parser.add_argument('--text_prompt_idx', type=int, default=4, help='text prompt index')
    text_prompts = ['A_person_half_kneel_with_one_leg_to_work_near_the_floor',
                    'A_person_half_squat_to_work_near_the_floor',
                    'A_person_move_a_box_from_left_to_right',
                    'A_person_raise_both_hands_above_his_head_and_keep_them_there',
                    'A_person_squat_to_carry_up_something']
    parser.add_argument('--text_prompts', type=list, default=text_prompts, help='text prompts')
    _args = parser.parse_args()
    _args.text_prompt = _args.text_prompts[_args.text_prompt_idx]
    return _args


if __name__ == '__main__':
    args = parse_args()

    ########################### Step 1: load npy motions into 3DSSPP loc files ###########################
    motion_smpl_folder = f'{args.motion_smpl_base_dir}\\{args.text_prompt}'
    search_string = args.search_string
    small_sample = args.small_sample
    # get all txt file with search_string in the filename
    motion_smpl_files = [filename for filename in os.listdir(motion_smpl_folder) if
                         filename.lower().endswith('.npy') and os.path.isfile(os.path.join(motion_smpl_folder, filename)) and search_string in filename]

    loc_file = f'{motion_smpl_folder}\\3DSSPP-all-{args.text_prompt}-{small_sample}.txt'  # intermediate output file
    last_frame_i = 0
    for motion_i, motion_smpl_file in enumerate(motion_smpl_files):
        if motion_i > small_sample: break
        print(f'processing {motion_i}:{motion_smpl_file}...')
        with open(os.path.join(motion_smpl_folder, motion_smpl_file), 'rb') as f:
            motion_smpl = np.load(f, allow_pickle=True)[None][0]
            global_orient = motion_smpl[:, :3]
            body_pose = motion_smpl[:, 3:]
        frame_no = body_pose.shape[0]
        smpl_object = SMPL(model_path=r'models\smpl\SMPL_NEUTRAL.pkl', batch_size=frame_no)
        body_pose = torch.tensor(body_pose, dtype=torch.float32)
        global_orient = torch.tensor(global_orient, dtype=torch.float32)
        smpl_output = smpl_object.forward(beta=np.zeros(10), body_pose=body_pose, global_orient=global_orient)
        joints = smpl_output.joints.detach().numpy()
        vertices = smpl_output.vertices.detach().numpy()
        faces = smpl_object.faces

        smpl_pose = SMPLPose()
        smpl_pose.load_smpl(joints, vertices, faces)
        smpl_pose.downsample(step=5)

        last_frame_i = smpl_pose.export_3DSSPP_batch(loc_file=loc_file, concatenate=last_frame_i, task_name=motion_smpl_file)

    ########################### Step 2: Run 3DSSPP ###########################
    # Get the initial modification time of the output file
    export_file = '3DSSPP_v7_cli/export/batchinput_export.txt'  # constant if using wrapper
    initial_mtime = os.stat(export_file).st_mtime
    loc_file = '../' + loc_file.replace('\\', '/')  # relative path to the loc file
    print(f"\n{'@' * 30} Subprocess start {'@' * 30}")
    subprocess.call(['bash', '3DSSPP-script.sh', loc_file], shell=True, cwd='3DSSPP_v7_cli/')
    # careful to look for errors messages in terminal for the subprocess, will not stop code
    wait_for_file_update(export_file, initial_mtime)  # Wait for the output file to be updated
    print(f"\n{'@' * 30} Subprocess end {'@' * 30}\n")

    # save a copy of the output file
    cp_export_file = loc_file.replace('.txt', '_export.txt')
    subprocess.call(['cp', export_file, cp_export_file], shell=True)

    ########################### Step 3: Analyze the output txt file ###########################
    # load file
    result = SSPPV7Output()
    result.load_file(export_file)
    result.cut_segment()

    result.show_category(subcategory='Summary')
    result.show_category()

    eval_keys = result.show_category(subcategory='Summary')[:-3]
    result.visualize_segment(result.all_segments, segment_eval_keys=eval_keys, verbose=True)

    # eval_keys = result.show_category(subcategory='Strength Value')[-6:-3] + result.show_category(subcategory='Strength Capability Percentile')[-6:-3]
    # result.visualize_segment(result.all_segments, segment_eval_keys=eval_keys, verbose=True)

    # eval_keys = result.show_category(subcategory='Strength Capability Percentile')[-6:-3]
    # result.visualize_segment(result.all_segments, segment_eval_keys=eval_keys, verbose=True)
    #
    # eval_keys = result.show_category(subcategory='Posture Angles')[-7:]
    # result.visualize_segment(result.all_segments, segment_eval_keys=eval_keys, verbose=True)

    eval_keys = result.show_category(subcategory='Strength Capability Percentile')

    ours = result.eval_segment(result.segments, eval_keys)
    result.eval_segment(result.segments[ours[-1]], eval_keys, verbose=True)
    baseline = result.eval_segment(result.baseline_segments, eval_keys, verbose=True)
    print("##################################################")
    print(f"text prompt: {args.text_prompt}")
    print(f"ours: {ours}")
    print(f"baseline: {baseline}")