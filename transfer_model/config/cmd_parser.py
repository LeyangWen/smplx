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

from __future__ import absolute_import
from __future__ import division

import sys
import os

import argparse
from loguru import logger

from omegaconf import OmegaConf
from .defaults import conf as default_conf


def parse_args(argv=None) -> OmegaConf:
    arg_formatter = argparse.ArgumentDefaultsHelpFormatter

    description = 'Model transfer script'
    parser = argparse.ArgumentParser(formatter_class=arg_formatter,
                                     description=description)

    parser.add_argument('--exp-cfg', type=str, dest='exp_cfg',
                        help='The configuration of the experiment')
    parser.add_argument('--exp-opts', default=[], dest='exp_opts',
                        nargs='*',
                        help='Command line arguments')

    parser.add_argument(
        "--batch-moshpp",
        action="store_true",
        help="Batch process moshpp output, will use args.motion-file as a directory",
    )
    parser.add_argument(
        "--batch-id",
        type=int,
    )
    parser.add_argument(
        "--overwrite-input-obj-folder", type=str, help="The path to the moshpp_batch input folder"
    )
    parser.add_argument(
        "--overwrite-output-folder", type=str, help="The path to the moshpp_batch output folder"
    )
    parser.add_argument('--wandb-project', default='smpl-smplx', help='wandb project name')
    parser.add_argument('--wandb-name', default='test', help='wandb run name')

    cmd_args = parser.parse_args()

    cfg = default_conf.copy()
    if cmd_args.exp_cfg:
        cfg.merge_with(OmegaConf.load(cmd_args.exp_cfg))
    if cmd_args.exp_opts:
        cfg.merge_with(OmegaConf.from_cli(cmd_args.exp_opts))
    return cfg, cmd_args  # exp_cfg, personal args
