#!/bin/bash -l
#SBATCH --job-name=smplx-merge
#SBATCH --output=output_slurm/merge_log.txt
#SBATCH --error=output_slurm/merge_error.txt
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=50g
#SBATCH --gres=gpu:1
#SBATCH --time=2:00:00
#SBATCH --account=shdpm0
#SBATCH --partition=spgpu


##### END preamble

my_job_header

conda activate soma3.7

module load clang/2022.1.2
module load gcc/10.3.0
module load gcc/13.2.0
module load intel/2022.1.2
module load boost/1.78.0
module load eigen tbb
module load blender
module list

cd transfer_model

python -u merge_output.py \
--gender male \
/nfs/turbo/coe-shdpm/leyang/VEHS-7M/Mesh/SMPL_obj_pkl/S01/Activity00_stageii/


