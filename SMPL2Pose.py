from smplx.body_models import SMPL, SMPLH, SMPLX, MANO, FLAME
import numpy as np
import matplotlib.pyplot as plt

smpl_object = SMPL(model_path=r'models\smpl\SMPL_NEUTRAL.pkl')
theta = np.zeros(72)
# theta[-3:] = 0.5
smpl_output = smpl_object.forward(beta=np.zeros(10), theta=theta)
joints_frame = smpl_output.joints[0].detach().numpy()
vertices_frame = smpl_output.vertices[0].detach().numpy()
joints_frame = joints_frame[:, [0, 2, 1]]
joints_frame[:, 1] = -joints_frame[:, 1]
vertices_frame = vertices_frame[:, [0, 2, 1]]
vertices_frame[:, 1] = -vertices_frame[:, 1]
if False:
    # plot vertices
    hand_center_z = joints_frame[22, 2]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for i, v in enumerate(vertices_frame):
        point_size = 1
        # if i%5!=0:
        #     continue
        # if v[2]>-0 or v[2]<-0.5:
        #     continue
        if v[0]<0.75 or v[2]>hand_center_z:
            continue
        ax.scatter(v[0], v[1], v[2], color='grey', s=point_size)
        ax.text(v[0], v[1], v[2], str(i), color='r')
    for j, joint in enumerate(joints_frame):
        ax.scatter(joint[0], joint[1], joint[2], color='g', s=0.5)
        ax.text(joint[0], joint[1], joint[2], str(j), color='b')
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

joints = np.array([joints_frame, joints_frame])
vertices = np.array([vertices_frame, vertices_frame])
weight = 70
height = 180
start_frame = 0
frame_no = 2
end_frame = frame_no
step = 1
fill = 0
loc = np.zeros((end_frame, 117))
# 1 - 3 Top Head Skin Surface
# 4 - 6 L. Head Skin Surface
# 7 - 9 R. Head Skin Surface
# 10 - 12 Head origin Virtual point
# 13 - 15 Nasion Skin Surface
# 16 - 18 Sight end Virtual point
# 19 - 21 C7/T1 Joint Center
# 22 - 24 Sternoclavicular Joint Joint Center
# 25 - 27 Suprasternale Skin Surface
# 28 - 30 L5/S1 Joint Center
# 31 - 33 PSIS Joint Center
# 34 - 36 L. Shoulder Joint Center
# 37 - 39 L. Acromion Skin Surface
# 40 - 42 L. Elbow Joint Center
# 43 - 45 L. Lat. Epicon. of Humer. Skin Surface
# 46 - 48 L. Wrist Joint Center
# 49 - 51 L. Grip Center Virtual point
# 52 - 54 L. Hand Skin Surface
# 55 - 57 R. Shoulder Joint Center
# 58 - 60 R. Acromion Skin Surface
# 61 - 63 R. Elbow Joint Center
# 64 - 66 R. Lat. Epicon. of Humer. Skin Surface
# 67 - 69 R. Wrist Joint Center
# 70 - 72 R. Grip Center Virtual point
# 73 - 75 R. Hand Skin Surface
# 76 - 79 L. Hip Joint Center
# 79 - 81 L. Knee Joint Center
# 82 - 84 L. Lat. Epicon. of Femur Skin Surface
# 85 - 87 L. Ankle Joint Center
# 88 - 90 L. Lateral Malleolus Skin Surface
# 91 - 93 L. Ball of Foot Virtual point
# 94 - 96 L. Metatarsalphalangeal Skin Surface
# 97 - 99 R. Hip Joint Center
# 100 - 102 R. Knee Joint Center
# 103 - 105 R. Lat. Epicon. of Femur Skin Surface
# 106 - 108 R. Ankle Joint Center
# 109 - 111 R. Lateral Malleolus Skin Surface
# 112 - 114 R. Ball of Foot Virtual point
# 115 - 117 R. Metatarsalphalangeal Skin Surface

loc[:, 0:3]         = vertices[:, 411, :]                   # 1 - 3 Top Head Skin Surface
loc[:, 3:6]         = joints[:, 28, :]                  # 4 - 6 L. Head Skin Surface
loc[:, 6:9]         = joints[:, 27, :]                  # 7 - 9 R. Head Skin Surface
loc[:, 9:12]        = (joints[:, 27, :]+joints[:, 28,:])/2                  # 10 - 12 Head origin Virtual point
loc[:, 12:15]       = joints[:, 24, :]                  # 13 - 15 Nasion Skin Surface
# # loc[:,15:18]      =                                   # 16 - 18 Sight end Virtual point
loc[:, 18:21]       = joints[:, 12, :]                  # 19 - 21 C7/T1 Joint Center
loc[:, 21:24]       = (joints[:, 14, :]+joints[:, 13,:])/2                  # 22 - 24 Sternoclavicular Joint Joint Center
# # loc[:,24:27]      =                                   # 25 - 27 Suprasternale Skin Surface
loc[:, 27:30]       = vertices[:, 3020, :]                  # 28 - 30 L5/S1 Joint Center
# loc[:, 30:33]       =                                   # 31 - 33 PSIS Joint Center
loc[:, 33:36]       = joints[:, 16, :]                                  # 34 - 36 L. Shoulder Joint Center
# # loc[:,36:39]      =                                   # 37 - 39 L. Acromion Skin Surface
loc[:, 39:42]       = joints[:, 18, :]                                  # 40 - 42 L. Elbow Joint Center
# loc[:, 42:45]       =                                   # 43 - 45 L. Lat. Epicon. of Humer. Skin Surface
loc[:, 45:48]       = joints[:, 20, :]                  # 46 - 48 L. Wrist Joint Center
loc[:, 48:51]       = joints[:, 22, :]                  # 49 - 51 L. Grip Center Virtual point
loc[:, 51:54]       = (joints[:, 22, :] + joints[:, 35, :] + np.mean(joints[:,36:40,:], axis=1))/2                   # 52 - 54 L. Hand Skin Surface
loc[:, 54:57]       = joints[:, 17, :]                  # 55 - 57 R. Shoulder Joint Center
# # loc[:,57:60]      =                                   # 58 - 60 R. Acromion Skin Surface
loc[:, 60:63]       = joints[:, 19, :]                  # 61 - 63 R. Elbow Joint Center
# loc[:, 63:66]       =                                   # 64 - 66 R. Lat. Epicon. of Humer. Skin Surface
loc[:, 66:69]       = joints[:, 21, :]                  # 67 - 69 R. Wrist Joint Center
loc[:, 69:72]       = joints[:, 23, :]                  # 70 - 72 R. Grip Center Virtual point
loc[:, 72:75]       = (joints[:, 23, :] + joints[:, 40, :] + np.mean(joints[:,41:45,:], axis=1))/2                                    # 73 - 75 R. Hand Skin Surface 2198
loc[:, 75:78]       = joints[:, 1, :]                  # 76 - 79 L. Hip Joint Center
loc[:, 78:81]       = joints[:, 4, :]                   # 79 - 81 L. Knee Joint Center
# loc[:, 81:84]       =                                   # 82 - 84 L. Lat. Epicon. of Femur Skin Surface
loc[:, 84:87]       = joints[:, 7, :]                   # 85 - 87 L. Ankle Joint Center
# loc[:, 87:90]       =                                   # 88 - 90 L. Lateral Malleolus Skin Surface
loc[:, 90:93]       = joints[:, 10, :]                  # 91 - 93 L. Ball of Foot Virtual point
# loc[:, 93:96]       =                                   # 94 - 96 L. Metatarsalphalangeal Skin Surface
loc[:, 96:99]       =  joints[:, 2, :]                  # 97 - 99 R. Hip Joint Center
loc[:, 99:102]      =  joints[:, 5, :]                  # 100 - 102 R. Knee Joint Center
# loc[:, 102:105]     =                                   # 103 - 105 R. Lat. Epicon. of Femur Skin Surface
loc[:, 105:108]     =  joints[:, 8, :]                  # 106 - 108 R. Ankle Joint Center
# loc[:, 108:111]     =                                   # 109 - 111 R. Lateral Malleolus Skin Surface
loc[:, 111:114]     =  joints[:, 11, :]                 # 112 - 114 R. Ball of Foot Virtual point
# loc[:, 114:117]     =                                   # 115 - 117 R. Metatarsalphalangeal Skin Surface

loc_file = f'experiment/3DSSPPBatch.txt'
# write as txt file
with open(loc_file, 'w') as f:
    f.write('3DSSPPBATCHFILE #\n')
    f.write('COM #\n')
    f.write('DES 1 "Task Name" "Analyst Name" "Comments" "Company" #\n')  # English is 0 and metric is 1
    for i, k in enumerate(np.arange(start_frame, end_frame, step)):
        joint_locations = np.array2string(loc[k], separator=' ', max_line_width=1000000, precision=3, suppress_small=True)[1:-1].replace('0. ', '0 ')
        # f.write('AUT 1 #\n')
        f.write('FRM ' + str(i + 1) + ' #\n')
        f.write(f'ANT 0 3 {height} {weight} #\n')  # male 0, female 1, self-set 3, height  , weight

        f.write(f'LOC {joint_locations} #\n')
        # f.write('HAN 15 -20 85 15 -15 80 #\n')
        # f.write('EXP #\n')
    # f.write('AUT 1 #\n')