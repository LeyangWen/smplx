import pickle
# import bpy
import argparse
import numpy as np
import ergo3d as eg
import xml.etree.ElementTree as ET
from smplx.body_models import SMPL, SMPLH, SMPLX, MANO, FLAME
import torch
from SMPLPose import SMPLPose

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str, default=r"W:/VEHS/VEHS-7M/SMPL/S01/Activity00_stageii.pkl")
    parser.add_argument('--camera_xcp', type=str, default=r"W:\VEHS\VEHS data collection round 3\processed\LeyangWen\FullCollection\Activity00.xcp")
    parser.add_argument('--file_type', type=str, default='mosh_pkl')
    parser.add_argument('--expression', type=str, default='pleasant')
    parser.add_argument('--frame', type=int, default=0)
    return parser.parse_args()


def get_expression(args):
    presets = {
        "pleasant": [0, .3, 0, -.892, 0, 0, 0, 0, -1.188, 0, .741, -2.83, 0, -1.48, 0, 0, 0, 0, 0, -.89, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, .89, 0, 0, 2.67],
        "happy": [0.9, 0, .741, -2, .27, -.593, -.29, 0, .333, 0, 1.037, -1, 0, .7, .296, 0, 0, -1.037, 0, 0, 0, 1.037, 0, 3],
        "excited": [-.593, .593, .7, -1.55, -.32, -1.186, -.43, -.14, -.26, -.88, 1, -.74, 1, -.593, 0, 0, 0, 0, 0, 0, -.593],
        "sad": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7.8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, -2, 0, 0, 0, 0, 0, 2, 2, -2, 1, 1.6, 2, 1.6],
        "frustrated": [0, 0, -1.33, 1.63, 0, -1.185, 2.519, 0, 0, -.593, -.444],
        "angry": [0, 0, -2.074, 1.185, 1.63, -1.78, 1.63, .444, .89, .74, -4, 1.63, -1.93, -2.37, -4],
    }
    return presets[args.expression]

def batch_load_from_xcp(xcp_filename):
    # Read the xcp file as xml
    with open(xcp_filename, 'r') as f:
        xml_string = f.read()
    root = ET.fromstring(xml_string)
    cameras = []
    for child in root:
        if child.attrib['DISPLAY_TYPE'] == 'VideoInputDevice:Blackfly S BFS-U3-23S3C':
            camera = eg.FLIR_Camera()
            camera.load_vicon_xcp(child)
            cameras.append(camera)
    return cameras


if __name__ == '__main__':
    args = parse_args()

    # # Step 1: read camera orientation & translation
    # cameras = batch_load_from_xcp(args.camera_xcp)
    # camera = cameras[4]
    # camera_orientation_quaternion = camera.ORIENTATION  # quaternion
    # # camera_orientation_quaternion = camera_orientation_quaternion * np.array([1, -1, -1, -1])  # get conjugate
    # camera_orientation_axis_angle = eg.Camera.axis_angle_from_quaternion(camera_orientation_quaternion)

    # Step 2: get SMPL pose
    with open(args.input_file, "rb") as f:
        data = pickle.load(f)

    pose = data['fullpose'][args.frame]
    trans_mosh = data['trans'][args.frame]
    trans = np.array([trans_mosh[0], trans_mosh[2], trans_mosh[1]])
        # np.array([0,0,0])  # x --> x in blender, y --> z in blender, z --> -y in blender
        #  - camera.POSITION
    #pose is the SMPL fullpose in axis angle representation
    global_orient = pose[:3]
    body_pose = pose[3:66]
    jaw_pose = pose[66:69]
    leye_pose = pose[69:72]
    reye_pose = pose[72:75]
    left_hand_pose = pose[75:120]
    right_hand_pose = pose[120:]

    # global_orient = eg.Camera.rotate_axis_angle_by_axis_angle(global_orient, np.array([-np.pi/2, 0, 0]))
    # global_orient = eg.Camera.rotate_axis_angle_by_axis_angle(global_orient, camera_orientation_axis_angle)
    # global_orient = eg.Camera.rotate_axis_angle_by_axis_angle(camera_orientation_axis_angle, global_orient)
    frame_no = 1
    smpl_object = SMPLX(model_path=r'models\smpl\SMPL_NEUTRAL.pkl', batch_size=frame_no)
    body_pose = torch.tensor(body_pose, dtype=torch.float32)
    global_orient = torch.tensor(global_orient, dtype=torch.float32)
    smpl_output = smpl_object.forward(beta=np.zeros(10), body_pose=body_pose, global_orient=global_orient)
    joints = smpl_output.joints.detach().numpy()
    vertices = smpl_output.vertices.detach().numpy()
    faces = smpl_object.faces

    smpl_pose = SMPLPose()
    smpl_pose.load_smpl(joints, vertices, faces)
    smpl_pose.downsample()

