from smplx.body_models import SMPL, SMPLH, SMPLX, MANO, FLAME
import numpy as np
import matplotlib.pyplot as plt
import torch
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import os
from utility import *
import imageio
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
import trimesh

class SMPLPose:
    def __init__(self):
        pass

    def load_smpl(self, joints, vertices, faces):
        # swap x and z axis, and flip y axis
        joints = joints[:, :, [0, 2, 1]]
        joints[:, :, 1] = -joints[:, :, 1]
        vertices = vertices[:, :, [0, 2, 1]]
        vertices[:, :, 1] = -vertices[:, :, 1]
        self.joints = joints
        self.vertices = vertices
        self.faces = faces
        self.frame_number = joints.shape[0]

    def downsample(self, step=3):
        self.joints = self.joints[::step]
        self.vertices = self.vertices[::step]
        self.frame_number = self.joints.shape[0]

    def plot_vertices_frame(self, frame=0, plot_joints=False, filename=False, render_mode='matplotlib'):
        joints_frame = self.joints[frame]
        vertices_frame = self.vertices[frame]
        if render_mode == 'matplotlib':
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            mesh = Poly3DCollection(vertices_frame[self.faces], alpha=0.1)
            face_color = (1.0, 1.0, 0.9)
            edge_color = (0, 0, 0)
            mesh.set_edgecolor(edge_color)
            mesh.set_facecolor(face_color)
            ax.add_collection3d(mesh)
            if plot_joints:
                ax.scatter(joints_frame[:, 0], joints_frame[:, 1], joints_frame[:, 2], alpha=0.2, color='r')
                # # vertices_index_list = np.arange(0000, 1000, 100)
                # # for vertices_index in vertices_index_list:
                # for vertices_index in range(vertices_frame.shape[0]):
                #     point_of_interest = vertices_frame[vertices_index]
                #     if point_of_interest[0] < 0.15 and point_of_interest[0] > -0.1:
                #         if point_of_interest[1] < 0.2 and point_of_interest[1] > 0.1:
                #             if point_of_interest[2] < -0.1 and point_of_interest[2] > -0.2:
                #                 ax.scatter(point_of_interest[0], point_of_interest[1], point_of_interest[2], alpha=1, color='b')
                #                 ax.text(point_of_interest[0], point_of_interest[1], point_of_interest[2], str(vertices_index), color='b')
            # lowwer view angle
            ax.view_init(elev=20, azim=-60)

            plot_range = 1.800
            pelvis_loc = self.joints[frame, 0, :]
            ax.set_xlim(pelvis_loc[0] - plot_range / 2, pelvis_loc[0] + plot_range / 2)
            ax.set_ylim(pelvis_loc[1] - plot_range / 2, pelvis_loc[1] + plot_range / 2)
            ax.set_zlim(pelvis_loc[2] - plot_range / 2, pelvis_loc[2] + plot_range / 2)
            ax.set_xlabel('X (m)')
            ax.set_ylabel('Y (m)')
            ax.set_zlabel('Z (m)')
            fig.tight_layout()
            if filename:
                plt.savefig(filename, dpi=250)
                plt.close(fig)
                return None
            else:
                plt.show()
                return fig, ax
        elif render_mode == 'open3d':
            #  right now render only
            mesh = trimesh.Trimesh(vertices=vertices_frame, faces=self.faces)

            # Extract vertices and faces
            trimesh_vertices = np.array(mesh.vertices)
            trimesh_faces = np.array(mesh.faces)

            # Create an Open3D TriangleMesh
            mesh_o3d = o3d.geometry.TriangleMesh()
            mesh_o3d.vertices = o3d.utility.Vector3dVector(trimesh_vertices)
            mesh_o3d.triangles = o3d.utility.Vector3iVector(trimesh_faces)

            # Generate numbers for vertices
            # vertex_numbers = [str(i) for i in range(len(trimesh_vertices))]
            colors = [[0, 0, 0] for _ in range(len(trimesh_vertices))]  # Black color for points

            # Create point clouds for the vertices
            point_cloud = o3d.geometry.PointCloud()
            point_cloud.points = o3d.utility.Vector3dVector(trimesh_vertices)
            # point_cloud.colors = o3d.utility.Vector3dVector(colors)

            # plot out joints
            joints_frame_expanded = joints_frame.copy()  # add more joints in +-xyz to make it look bigger in the point cloud
            for x_offset in range(-2, 3):
                for y_offset in range(-2, 3):
                    for z_offset in range(-2, 3):
                        joints_frame_expanded = np.concatenate((joints_frame_expanded, joints_frame + np.array([x_offset, y_offset, z_offset])*0.005), axis=0)
            joint_point_cloud = o3d.geometry.PointCloud()
            joint_point_cloud.points = o3d.utility.Vector3dVector(joints_frame_expanded)
            joint_colors = [[1, 0, 0] for _ in range(len(joints_frame_expanded))]  # Black color for points
            joint_point_cloud.colors = o3d.utility.Vector3dVector(joint_colors)
            # point size



            app = gui.Application.instance
            app.initialize()
            # Create visualizer
            vis = o3d.visualization.O3DVisualizer("Open3D", 1024, 768)

            for i, vertex in enumerate(trimesh_vertices):
                if vertex[1] < trimesh_vertices[4927][1]:  # > front side
                    continue
                vis.add_3d_label(vertex, str(i))
            for i, vertex in enumerate(joints_frame):
                vis.add_3d_label(vertex, str(i))
            # Add the mesh and point cloud to the visualizer
            #
            vis.add_geometry("points", point_cloud)
            vis.add_geometry("joints", joint_point_cloud)
            # vis.add_geometry("mesh", mesh_o3d)

            vis.reset_camera_to_default()
            app.add_window(vis)
            app.run()

    def plot_vertices(self, foldername=False, make_gif=False, fps=30):
        if foldername:
            create_dir(foldername)
        for i in range(self.frame_number):
            print(f'plotting frame {i}/{self.frame_number} in {foldername}...', end='\r')
            filename = foldername if not foldername else os.path.join(foldername, f'{i:05d}.png')
            self.plot_vertices_frame(frame=i, filename=filename)

        if foldername and make_gif:
            images = []
            print(f'making gif in {foldername}...')
            image_files = [filename for filename in os.listdir(foldername) if filename.lower().endswith('.png') and os.path.isfile(os.path.join(foldername, filename))]
            image_files.sort()
            for image_file in image_files:
                # Load each image and append it to the images list
                image_path = os.path.join(foldername, image_file)
                images.append(imageio.imread(image_path))

            # Save images as a GIF
            output_filename = os.path.join(foldername, '00000000_merged.gif')
            imageio.mimsave(output_filename, images, duration=self.frame_number/fps, loop=100)

    def export_3DSSPP_batch(self, loc_file=f'experiment/3DSSPPBatch.txt', weigh_height=None, start_frame=0, end_frame=0, concatenate=0, task_name=''):
        """
        :param loc_file: 3DSSPP batch file
        :param weigh_height: [weight, height], if None, use default 50th percentile
        :param concatenate: False or int, if not false, append to the end of the file at int(concatenate) frame
        """
        '''
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
        '''
        if weigh_height is None:
            weigh_height = [80, 180]
        if end_frame == 0:
            end_frame = self.frame_number
        if start_frame != 0:
            raise NotImplementedError

        step = 1
        loc = np.zeros((end_frame, 117))
        loc[:, 0:3] = self.vertices[:, 411, :]  # 1 - 3 Top Head Skin Surface
        loc[:, 3:6] = self.joints[:, 28, :]  # 4 - 6 L. Head Skin Surface
        loc[:, 6:9] = self.joints[:, 27, :]  # 7 - 9 R. Head Skin Surface
        loc[:, 9:12] = (self.joints[:, 27, :] + self.joints[:, 28, :]) / 2  # 10 - 12 Head origin Virtual point
        loc[:, 12:15] = (self.joints[:, 25, :] + self.joints[:, 26, :]) / 2  # 13 - 15 Nasion Skin Surface
        # # loc[:,15:18]      =                                   # 16 - 18 Sight end Virtual point
        loc[:, 18:21] = self.joints[:, 12, :]  # 19 - 21 C7/T1 Joint Center
        loc[:, 21:24] = (self.vertices[:, 2812, :] + (self.joints[:, 13, :] + self.joints[:, 14, :]) / 2) / 2  # 22 - 24 Sternoclavicular Joint Joint Center
        loc[:, 24:27] = self.vertices[:, 2812, :]  # 25 - 27 Suprasternale Skin Surface
        loc[:, 27:30] = self.joints[:, 0, :]*0.5 + self.vertices[:, 3158, :]*0.5  # 28 - 30 L5/S1 Joint Center
        loc[:, 30:33] = self.joints[:, 0, :]*0.1 + self.vertices[:, 3159, :]*0.9  # 31 - 33 PSIS Joint Center
        loc[:, 33:36] = self.joints[:, 16, :]  # 34 - 36 L. Shoulder Joint Center
        loc[:, 36:39] = self.vertices[:, 1821, :]  # 37 - 39 L. Acromion Skin Surface
        loc[:, 39:42] = self.joints[:, 18, :]  # 40 - 42 L. Elbow Joint Center
        loc[:, 42:45] = self.vertices[:, 1700, :]  # 43 - 45 L. Lat. Epicon. of Humer. Skin Surface
        loc[:, 45:48] = self.joints[:, 20, :]  # 46 - 48 L. Wrist Joint Center
        loc[:, 48:51] = (self.joints[:, 22, :] + np.mean(self.joints[:, 36:40, :], axis=1)) / 2  # 49 - 51 L. Grip Center Virtual point
        loc[:, 51:54] = self.joints[:, 22, :]  # 52 - 54 L. Hand Skin Surface
        loc[:, 54:57] = self.joints[:, 17, :]  # 55 - 57 R. Shoulder Joint Center
        loc[:, 57:60] = self.vertices[:, 5282, :]  # 58 - 60 R. Acromion Skin Surface
        loc[:, 60:63] = self.joints[:, 19, :]  # 61 - 63 R. Elbow Joint Center
        loc[:, 63:66] = self.vertices[:, 5171, :]  # 64 - 66 R. Lat. Epicon. of Humer. Skin Surface
        loc[:, 66:69] = self.joints[:, 21, :]  # 67 - 69 R. Wrist Joint Center
        loc[:, 69:72] = (self.joints[:, 23, :] + np.mean(self.joints[:, 41:45, :], axis=1)) / 2  # 70 - 72 R. Grip Center Virtual point
        loc[:, 72:75] = self.joints[:, 23, :]  # 73 - 75 R. Hand Skin Surface 2198
        loc[:, 75:78] = self.joints[:, 1, :] * 0.8 + self.joints[:, 4, :] * 0.2  # 76 - 79 L. Hip Joint Center
        loc[:, 78:81] = self.joints[:, 4, :]  # 79 - 81 L. Knee Joint Center
        loc[:, 81:84] = self.vertices[:, 1010, :]  # 82 - 84 L. Lat. Epicon. of Femur Skin Surface
        loc[:, 84:87] = self.joints[:, 7, :]  # 85 - 87 L. Ankle Joint Center
        # loc[:, 87:90]       =                                   # 88 - 90 L. Lateral Malleolus Skin Surface
        loc[:, 90:93] = self.joints[:, 10, :]  # 91 - 93 L. Ball of Foot Virtual point
        # loc[:, 93:96]       =                                   # 94 - 96 L. Metatarsalphalangeal Skin Surface
        loc[:, 96:99] = self.joints[:, 2, :] * 0.8 + self.joints[:, 5, :] * 0.2  # 97 - 99 R. Hip Joint Center
        loc[:, 99:102] = self.joints[:, 5, :]  # 100 - 102 R. Knee Joint Center
        loc[:, 102:105] = self.vertices[:, 4539, :]  # 103 - 105 R. Lat. Epicon. of Femur Skin Surface
        loc[:, 105:108] = self.joints[:, 8, :]  # 106 - 108 R. Ankle Joint Center
        # loc[:, 108:111]     =                                   # 109 - 111 R. Lateral Malleolus Skin Surface
        loc[:, 111:114] = self.joints[:, 11, :]  # 112 - 114 R. Ball of Foot Virtual point
        # loc[:, 114:117]     =                                   # 115 - 117 R. Metatarsalphalangeal Skin Surface

        assert isinstance(concatenate, int) if concatenate else True
        mode = 'a' if concatenate else 'w'
        # write as txt file
        with open(loc_file, mode) as f:
            f.write('3DSSPPBATCHFILE #\n') if concatenate == 0 or concatenate is False else f.write('\n')  # if first line, write header, else write new line
            # f.write('COM #\n')  # comment
            f.write(f'DES 1 "Task-{task_name}-{concatenate}" "Analyst Name" "Comments" "Company" #\n')  # English is 0 and metric is 1
            for i, k in enumerate(np.arange(start_frame, end_frame, step)):
                joint_locations = np.array2string(loc[k], separator=' ', max_line_width=1000000, precision=3, suppress_small=True)[1:-1].replace('0. ', '0 ')
                frame_i = i+int(concatenate)+1
                f.write('FRM ' + str(frame_i) + ' #\n')
                if weigh_height[0] is None:
                    f.write(f'ANT 0 1  #\n')
                else:
                    f.write(f'ANT 0 3 {weigh_height[1]} {weigh_height[0]} #\n')  # 2nd int: male 0, female 1; 3rd int: 95th is 0, 50th is 1, and 5th is 2, self-set 3 - followed by height , weight
                # f.write(f'HAN 0 0 0 0 0 0 #\n')  # this seems to be causing bug in 3DSSPP after 800+ commands
                f.write(f'LOC {joint_locations} #\n')
                # f.write('HAN 15 -20 85 15 -15 80 #\n')
                # f.write('EXP #\n')
                f.write('AUT 1 #')
        return frame_i



if __name__ == '__main__':
    case = 0
    if case == 0:  # get all 50 results in one txt for all text prompt
        motion_smpl_folder_base = r'experiment\text2pose-20231113T194712Z-001\text2pose'
        text_prompts = ['A_person_half_kneel_with_one_leg_to_work_near_the_floor',
                        'A_person_half_squat_to_work_near_the_floor',
                        'A_person_move_a_box_from_left_to_right',
                        'A_person_raise_both_hands_above_his_head_and_keep_them_there',
                        'A_person_squat_to_carry_up_something']
        text_prompt = text_prompts[2]
        motion_smpl_folder = f'{motion_smpl_folder_base}\\{text_prompt}'

        search_string = "smpl_pose_72"
        # get all txt file with search_string in the filename
        motion_smpl_files = [filename for filename in os.listdir(motion_smpl_folder) if filename.lower().endswith('.npy') and os.path.isfile(os.path.join(motion_smpl_folder, filename)) and search_string in filename]

        loc_file = f'{motion_smpl_folder}\\3DSSPP-all-{text_prompt}.txt'
        last_frame_i = 0
        # for motion_i in range(30):
        # for motion_i in [15, 17, 23, 24, 42, 48]:
            # motion_i = 42
            # motion_smpl_file = f'{motion_smpl_folder}\smpl_pose_72_{motion_i}.npy'
            # motion_smpl_file = f'G:\My Drive\DPM\\temp\smpl_pose_72_{motion_i}.npy'
            # motion_smpl_file = r'C:\Users\wenleyan1\Downloads\Baseline_smpl_pose_72.npy'
        for motion_i, motion_smpl_file in enumerate(motion_smpl_files):
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
            # smpl_pose.plot_vertices_frame(frame=0, plot_joints=True, render_mode='open3d')
            # # smpl_pose.plot_vertices(foldername=f'{motion_smpl_folder}\smpl_pose_72_{motion_i}', make_gif=True, fps=30)
    # elif case == 2:  # get one individual result

