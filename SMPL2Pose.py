from smplx.body_models import SMPL, SMPLH, SMPLX, MANO, FLAME
import numpy as np
import matplotlib.pyplot as plt
import torch

motion_smpl_file = r'G:\My Drive\DPM\temp\smpl_pose_72_8.npy'
# motion_smpl_file = r'G:\My Drive\DPM\temp\motion_0.npy'
with open(motion_smpl_file, 'rb') as f:
    motion_smpl = np.load(f, allow_pickle=True)[None][0]
    global_orient = motion_smpl[:, :3]
    body_pose = motion_smpl[:, 3:]
    # motion_smpl = motion_smpl.reshape(-1, 24, 3)
frame_no = body_pose.shape[0]
smpl_object = SMPL(model_path=r'models\smpl\SMPL_NEUTRAL.pkl', batch_size=frame_no)
# theta = np.zeros(72)
# theta[-3:] = 0.5
# convert to a tensor
theta = torch.tensor(body_pose, dtype=torch.float32)
global_orient = torch.tensor(global_orient, dtype=torch.float32)
smpl_output = smpl_object.forward(beta=np.zeros(10), body_pose=theta, global_orient=global_orient)
joints = smpl_output.joints.detach().numpy()
vertices = smpl_output.vertices.detach().numpy()
joints = joints[:, :, [0, 2, 1]]
joints[:, :, 1] = -joints[:, :, 1]
vertices = vertices[:, :, [0, 2, 1]]
vertices[:, :, 1] = -vertices[:, :, 1]



if True:
    # plot vertices
    view_frame =   0
    joints_frame = joints[view_frame]
    vertices_frame = vertices[view_frame]

    hand_center_z = joints_frame[22, 2]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for i, v in enumerate(vertices_frame):
        point_size = 1
        if i%10!=0:
            continue
        # if v[2]>-0 or v[2]<-0.5:
        #     continue
        # if v[0]<0.75 or v[2]>hand_center_z:
        #     continue
        ax.scatter(v[0], v[1], v[2], color='grey', s=point_size)
        # ax.text(v[0], v[1], v[2], str(i), color='r')
    for j, joint in enumerate(joints_frame):
        ax.scatter(joint[0], joint[1], joint[2], color='g', s=10)
        # ax.text(joint[0], joint[1], joint[2], str(j), color='b')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    lim_max = np.max(vertices_frame)
    lim_min = np.min(vertices_frame)
    ax.set_xlim(lim_min, lim_max)
    ax.set_ylim(lim_min, lim_max)
    ax.set_zlim(lim_min, lim_max)
    fig.tight_layout()
    plt.show()


