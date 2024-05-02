# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2020 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# Contact: Vassilis Choutas, vassilis.choutas@tuebingen.mpg.de

import os
import os.path as osp
import sys
import pickle

import numpy as np
import open3d as o3d
import torch
from loguru import logger
from tqdm import tqdm
import wandb
import time

from smplx import build_layer

from .config import parse_args
from .data import build_dataloader
from .transfer_model import run_fitting
from .utils import read_deformation_transfer, np_mesh_to_o3d


def main(exp_cfg) -> None:


    if torch.cuda.is_available() and exp_cfg["use_cuda"]:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
        if exp_cfg["use_cuda"]:
            if input("use_cuda=True and GPU is not available, using CPU instead,"
                     " would you like to continue? (y/n)") != "y":
                sys.exit(3)
    print("*" * 20, "device:", device, "*" * 20)

    logger.remove()
    logger.add(
        lambda x: tqdm.write(x, end=''), level=exp_cfg.logger_level.upper(),
        colorize=True)

    output_folder = osp.expanduser(osp.expandvars(exp_cfg.output_folder))
    logger.info(f'Saving output to: {output_folder}')
    os.makedirs(output_folder, exist_ok=True)

    model_path = exp_cfg.body_model.folder
    body_model = build_layer(model_path, **exp_cfg.body_model)
    logger.info(body_model)
    body_model = body_model.to(device=device)

    deformation_transfer_path = exp_cfg.get('deformation_transfer_path', '')
    def_matrix = read_deformation_transfer(
        deformation_transfer_path, device=device)

    # Read mask for valid vertex ids
    mask_ids_fname = osp.expandvars(exp_cfg.mask_ids_fname)
    mask_ids = None
    if osp.exists(mask_ids_fname):
        logger.info(f'Loading mask ids from: {mask_ids_fname}')
        mask_ids = np.load(mask_ids_fname)
        mask_ids = torch.from_numpy(mask_ids).to(device=device)
    else:
        logger.warning(f'Mask ids fname not found: {mask_ids_fname}')

    data_obj_dict = build_dataloader(exp_cfg)

    dataloader = data_obj_dict['dataloader']

    subject_start_time = time.time()
    for ii, batch in enumerate(tqdm(dataloader)):
        for key in batch:
            if torch.is_tensor(batch[key]):
                batch[key] = batch[key].to(device=device)
        var_dict = run_fitting(
            exp_cfg, batch, body_model, def_matrix, mask_ids)
        paths = batch['paths']
        subject_cumulative_time = time.time() - subject_start_time
        wandb.log({'frames': exp_cfg.batch_size*ii, 'batch': ii, 'subject_cumulative_time_hr': subject_cumulative_time/3600})

        for ii, path in enumerate(paths):
            _, fname = osp.split(path)
            output_path = osp.join(
                output_folder, f'{osp.splitext(fname)[0]}.pkl')
            with open(output_path, 'wb') as f:
                pickle.dump(var_dict, f)

            output_path = osp.join(
                output_folder, f'{osp.splitext(fname)[0]}.obj')
            mesh = np_mesh_to_o3d(
                var_dict['vertices'][ii], var_dict['faces'])
            o3d.io.write_triangle_mesh(output_path, mesh)


if __name__ == '__main__':
    exp_cfg, args = parse_args()
    wandb.init(project=args.wandb_project, name=args.wandb_name, config=args)  # Initialize a new run
    start_time = time.time()
    current_time = time.time()

    male_subject_list = ['S01', 'S02', 'S04', 'S07', 'S08']
    female_subject_list = ['S03', 'S05', 'S06', 'S09', 'S10']
    if args.batch_moshpp:
        for root, dirs, files in os.walk(args.overwrite_input_obj_folder):
            if not root[-7:] == "stageii":
                continue
            activity_name = root.split('/')[-1]  # "Activity00_stageii"
            subject_name = root.split('/')[-2]  # "S01"
            subject_idx = int(subject_name[-2:])  # ".../SMPLX_obj/S01/Activity00_stageii/" --> "S01"  --> "01" --> 1
            if args.batch_id is not None and args.batch_id != subject_idx:
                continue
            print(f"Processing {args.batch_id} - {root}")

            exp_cfg.datasets.mesh_folder.data_folder = root
            exp_cfg.output_folder = os.path.join(args.overwrite_output_folder, subject_name, activity_name)
            if subject_name in male_subject_list:
                exp_cfg.body_model.gender = "male"
            elif subject_name in female_subject_list:
                exp_cfg.body_model.gender = "female"
            else:
                exp_cfg.body_model.gender = "neutral"
            print(f"Subject: {subject_name} is {exp_cfg.body_model.gender}")
            print(f"Output folder: {exp_cfg.output_folder}")
            main(exp_cfg)

            print("#"*20, f"Finished processing {args.batch_id} - {root}", "#"*20)
            iteration_time = time.time() - current_time
            culmulative_time = time.time() - start_time
            current_time = time.time()
            wandb.log({'subject_idx': subject_idx, 'iteration_time': iteration_time, 'culmulative_time': culmulative_time})

    else:
        main(exp_cfg)
    wandb.finish()

