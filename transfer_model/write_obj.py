import os.path
import os.path as osp
import argparse

import numpy as np
import torch


import trimesh

import smplx
from smplx.joint_names import Body

from tqdm.auto import tqdm, trange

from pathlib import Path
# import wandb

def main(
    model_folder,
    motion_file,
    output_folder,
    model_type="smplh",
    ext="npz",
    gender="neutral",
    plot_joints=False,
    num_betas=10,
    sample_expression=True,
    num_expression_coeffs=10,
    use_face_contour=False,
    verbose=True
):
    output_folder = Path(output_folder)
    assert output_folder.exists()

    # open motion file
    if motion_file.endswith(".npz"):
        motion = np.load(motion_file, allow_pickle=True)
    elif motion_file.endswith(".pkl"):  # output from moshpp
        motion = np.load(motion_file, allow_pickle=True)
        motion['poses'] = motion['fullpose']
        motion['marker_labels'] = np.array(motion['latent_labels'])
    else:
        raise ValueError("Unsupported file type")

    for k, v in motion.items():
        if type(v) is float:
            print(k, v)
        elif type(v) is dict:
            print(k, type(v))
        elif type(v) is list:
            print(k, len(v))
        else:
            print(k, v.shape)

    if "betas" in motion:
        betas = motion["betas"]
    else:
        betas = np.zeros((num_betas,))
    num_betas = len(betas)
    # don't know where this is documented but it's from this part of amass
    # https://github.com/nghorbani/amass/blob/master/src/amass/data/prepare_data.py#L39-L40
    # gdr2num = {'male':-1, 'neutral':0, 'female':1}
    # gdr2num_rev = {v:k for k,v in gdr2num.items()}
    if "gender" in motion:
        gender = str(motion["gender"])
    else:
        gender = gender

    print(gender)

    print(num_betas)
    model = smplx.create(
        model_folder,
        model_type=model_type,
        gender=gender,
        use_face_contour=use_face_contour,
        num_betas=num_betas,
        num_expression_coeffs=num_expression_coeffs,
        use_pca=False,
        ext=ext,
    )

    betas, expression = torch.tensor(betas).float(), None
    betas = betas.unsqueeze(0)[:, : model.num_betas]
    if "poses" in motion:
        poses = torch.tensor(motion["poses"]).float()
        n = poses.shape[0]
    elif "smpl_poses" in motion:
        poses = motion["smpl_poses"]
        n = poses.shape[0]
        if model_type == "smplh":
            poses = np.stack(
                [Body.from_smpl(p.reshape(-1, 3)).as_smplh() for p in poses]
            )
        poses = torch.tensor(poses.reshape(n, -1)).float()
    global_orient = poses[:, :3]
    if model_type == "smplh":
        body_pose = poses[:, 3:66]
        left_hand_pose = poses[:, 66:111]
        right_hand_pose = poses[:, 111:156]
    elif model_type == "smplx":
        body_pose = poses[:, 3:66]
        jaw_pose = poses[:, 66:69]
        leye_pose = poses[:, 69:72]
        reye_pose = poses[:, 72:75]
        left_hand_pose = poses[:, 75:120]
        right_hand_pose = poses[:, 120:]
    else:
        body_pose = poses[:, 3:]
        left_hand_pose = np.zeros((n, 3))
        right_hand_pose = np.zeros((n, 3))
    # if sample_expression:
    #     expression = torch.randn(
    #         [1, model.num_expression_coeffs], dtype=torch.float32)

    # print(expression)
    # print(betas.shape, body_pose.shape, expression.shape)
    for pose_idx in trange(body_pose.size(0)):
        pose_idx = [pose_idx]
        # output = model(betas=betas, # expression=expression,
        #                return_verts=True)
        output = model(
            betas=betas,
            global_orient=global_orient[pose_idx],
            body_pose=body_pose[pose_idx],
            left_hand_pose=left_hand_pose[pose_idx],
            right_hand_pose=right_hand_pose[pose_idx],
            # expression=expression,
            return_verts=True,
        )
        vertices = output.vertices.detach().cpu().numpy().squeeze()
        joints = output.joints.detach().cpu().numpy().squeeze()

        vertex_colors = np.ones([vertices.shape[0], 4]) * [0.3, 0.3, 0.3, 0.8]
        # process=False to avoid creating a new mesh
        tri_mesh = trimesh.Trimesh(
            vertices, model.faces, vertex_colors=vertex_colors, process=False
        )

        output_path = output_folder / "{0:04d}.obj".format(pose_idx[0])
        tri_mesh.export(str(output_path))

        if pose_idx[0] == 0 and args.verbose:
            import pyrender
            print("displaying first pose, exit window to continue processing")
            mesh = pyrender.Mesh.from_trimesh(tri_mesh)

            scene = pyrender.Scene()
            scene.add(mesh)

            if plot_joints:
                sm = trimesh.creation.uv_sphere(radius=0.005)
                sm.visual.vertex_colors = [0.9, 0.1, 0.1, 1.0]
                tfs = np.tile(np.eye(4), (len(joints), 1, 1))
                tfs[:, :3, 3] = joints
                joints_pcl = pyrender.Mesh.from_trimesh(sm, poses=tfs)
                scene.add(joints_pcl)

            pyrender.Viewer(scene, use_raymond_lighting=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SMPL-X Demo")

    parser.add_argument(
        "--model-folder", required=True, type=str, help="The path to the model folder"
    )
    parser.add_argument(
        "--motion-file",
        required=True,
        type=str,
        help="The path to the motion file to process",
    )
    parser.add_argument(
        "--output-folder", required=True, type=str, help="The path to the output folder"
    )
    parser.add_argument(
        "--model-type",
        default="smplh",
        type=str,
        choices=["smpl", "smplh", "smplx", "mano", "flame"],
        help="The type of model to load",
    )
    parser.add_argument(
        "--num-expression-coeffs",
        default=10,
        type=int,
        dest="num_expression_coeffs",
        help="Number of expression coefficients.",
    )
    parser.add_argument(
        "--ext", type=str, default="npz", help="Which extension to use for loading"
    )
    parser.add_argument(
        "--sample-expression",
        default=True,
        dest="sample_expression",
        type=lambda arg: arg.lower() in ["true", "1"],
        help="Sample a random expression",
    )
    parser.add_argument(
        "--use-face-contour",
        default=False,
        type=lambda arg: arg.lower() in ["true", "1"],
        help="Compute the contour of the face",
    )
    parser.add_argument(
        "--batch-moshpp",
        action="store_true",
        help="Batch process moshpp output, will use args.motion-file as a directory",
    )
    parser.add_argument(
        "--batch-id",
        type=int,
    )

    args = parser.parse_args()

    def resolve(path):
        return osp.expanduser(osp.expandvars(path))

    model_folder = resolve(args.model_folder)
    motion_file = resolve(args.motion_file)
    output_folder = resolve(args.output_folder)
    model_type = args.model_type
    ext = args.ext
    num_expression_coeffs = args.num_expression_coeffs
    sample_expression = args.sample_expression

    if args.batch_moshpp:
        for root, dirs, files in os.walk(motion_file):
            dirs.sort()  # Sort directories in-place
            files.sort(key=str.lower)  # Sort files in-place
            for file in files:
                if file.endswith('.pkl') and 'stageii' in file:
                    if args.batch_id is not None and args.batch_id != int(dirs[-2:]):  # "S01"  --> "01" --> 1
                        continue
                        print(f"Processing {args.batch_id}")

                    # Determine gender from mosh output
                    female_dir = os.path.join(root, "female_stagei.json")
                    male_dir = os.path.join(root, "male_stagei.json")
                    if os.path.exists(female_dir):
                        gender = "female"
                    elif os.path.exists(male_dir):
                        gender = "male"
                    else:
                        print("*"*20, "Warning: No gender files detected", "*"*20)
                        gender = "neutral"
                    print(f"Setting gender to: {gender}")

                    input_file = os.path.join(root, file)
                    output_folder = os.path.join(output_folder, root.split('/')[-1], file.split('.')[0])
                    print(f"Processing {input_file}...")
                    print(f"Output folder: {output_folder}")
                    if not os.path.exists(output_folder):
                        os.makedirs(output_folder)

                    main(
                        model_folder,
                        input_file,
                        output_folder,
                        model_type,
                        ext=ext,
                        gender=gender,
                        sample_expression=sample_expression,
                        use_face_contour=args.use_face_contour,
                        verbose=False
                    )
    else:
        main(
            model_folder,
            motion_file,
            output_folder,
            model_type,
            ext=ext,
            sample_expression=sample_expression,
            use_face_contour=args.use_face_contour,
        )
