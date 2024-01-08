import argparse
import subprocess
import os
import time

from SMPLPose import SMPLPose
from SSPPOutput import SSPPV7Output



def parse_args():
    parser = argparse.ArgumentParser(description='SMPLPose')
    parser.add_argument('--small_sample', type=int, default=2, help='only process the first 2 files for testing')
    parser.add_argument('--input_dir', type=str, default='3DSSPP_v7_cli/', help='input directory')
    parser.add_argument('--input_file', type=str, default='example_input_batch.txt', help='input batch file')
    parser.add_argument('--output_dir', type=str, default='path_to_your_output_dir', help='output directory')
    parser.add_argument('--output_file', type=str, default='path_to_your_output_file', help='output file')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()


    # Step 1: load SMPLPose object
    # smpl_pose = SMPLPose()

    # Get the initial modification time of the output file
    output_file = '3DSSPP_v7_cli/export/batchinput_export.txt'
    initial_mtime = os.stat(output_file).st_mtime

    # Step 2: Run 3DSSPP
    subprocess.call(['bash', '3DSSPP-script.sh', 'example_input_batch.txt'], shell=True, cwd='3DSSPP_v7_cli/')

    # Step 3: Wait for the output file to be updated
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

    # Step 4: Analyze the output
    # ssp_output = SSPPV7Output()