import argparse
import subprocess
import os
import time

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
                    time.sleep(1)
                else:
                    print("Finished writing the output file", end="\r")
                    time.sleep(1)
                    break
            break



def parse_args():
    parser = argparse.ArgumentParser(description='SMPLPose')
    parser.add_argument('--small_sample', type=int, default=2, help='only process the first 2 files for testing')
    parser.add_argument('--input_dir', type=str, default='3DSSPP_v7_cli/', help='input text file directory')
    parser.add_argument('--input_file', type=str, default='example_input_batch.txt', help='input text file')
    parser.add_argument('--output_dir', type=str, default='path_to_your_output_dir', help='output directory')
    parser.add_argument('--output_file', type=str, default='path_to_your_output_file', help='output file')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()


    ########################### Step 1: load SMPLPose object ###########################
    motion_smpl_folder_base = r'experiment\text2pose-20231113T194712Z-001\text2pose'
    text_prompts = ['A_person_half_kneel_with_one_leg_to_work_near_the_floor',
                    'A_person_half_squat_to_work_near_the_floor',
                    'A_person_move_a_box_from_left_to_right',
                    'A_person_raise_both_hands_above_his_head_and_keep_them_there',
                    'A_person_squat_to_carry_up_something']
    text_prompt = text_prompts[2]
    motion_smpl_folder = f'{motion_smpl_folder_base}\\{text_prompt}'

    search_string = "smpl_pose_72"
    small_sample = 2  # only process the first 2 files, todo: remove when actually running
    # get all txt file with search_string in the filename
    motion_smpl_files = [filename for filename in os.listdir(motion_smpl_folder) if
                         filename.lower().endswith('.npy') and os.path.isfile(os.path.join(motion_smpl_folder, filename)) and search_string in filename]

    loc_file = f'{motion_smpl_folder}\\3DSSPP-all-{text_prompt}-{small_sample}.txt'
    last_frame_i = 0
    # for motion_i in range(30):
    # for motion_i in [15, 17, 23, 24, 42, 48]:
    # motion_i = 42
    # motion_smpl_file = f'{motion_smpl_folder}\smpl_pose_72_{motion_i}.npy'
    # motion_smpl_file = f'G:\My Drive\DPM\\temp\smpl_pose_72_{motion_i}.npy'
    # motion_smpl_file = r'C:\Users\wenleyan1\Downloads\Baseline_smpl_pose_72.npy'
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
        smpl_pose.downsample()

        last_frame_i = smpl_pose.export_3DSSPP_batch(loc_file=loc_file, concatenate=last_frame_i, task_name=motion_smpl_file)

    # Get the initial modification time of the output file
    output_file = '3DSSPP_v7_cli/export/batchinput_export.txt'
    initial_mtime = os.stat(output_file).st_mtime

    ########################### Step 2: Run 3DSSPP ###########################
    subprocess.call(['bash', '3DSSPP-script.sh', 'example_input_batch.txt'], shell=True, cwd='3DSSPP_v7_cli/')
    wait_for_file_update(output_file, initial_mtime)  # Wait for the output file to be updated


    ########################### Step 3: Analyze the output ###########################
    # load file
    input_3DSSPP_folder = r'experiment'
    input_3DSSPP_files = ['wrapper_multi_task.txt', 'wrapper_single_task.txt', 'test.txt']
    input_3DSSPP_file = input_3DSSPP_files[0]

    input_3DSSPP_file = r"text2pose-20231113T194712Z-001\text2pose\A_person_squat_to_carry_up_something\3DSSPP-all-A_person_squat_to_carry_up_something-2_export.txt"
    result = SSPPV7Output()
    result.load_file(os.path.join(input_3DSSPP_folder, input_3DSSPP_file))
    result.cut_segment()

    eval_keys = result.show_category(subcategory='Strength Capability Percentile')[-6:-3]
    result.visualize_segment(result.all_segments, segment_eval_keys=eval_keys, verbose=True)