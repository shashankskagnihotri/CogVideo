#!/bin/bash

#SBATCH --job-name=safety_cogvideox
#SBATCH --output=slurm/safety/initial_%A_%a.out
#SBATCH --error=slurm/safety/initial_%A_%a.err
#SBATCH --cpus-per-task=16
#SBATCH --mem=350Gb
#SBATCH --time=119:59:59
#SBATCH --array=0-7
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu-vram-94gb
#SBATCH --mail-type=ALL
#SBATCH --mail-user=shashank.agnihotri@uni-mannheim.de

echo "Started at $(date)";
echo "Running job: $SLURM_JOB_NAME array id: $SLURM_ARRAY_TASK_ID using $SLURM_JOB_CPUS_PER_NODE cpus per node with given JID $SLURM_JOB_ID on queue $SLURM_JOB_PARTITION";

start=`date +%s`

python inference/safe_check_cli_demo.py --model_path THUDM/CogVideoX1.5-5b --generate_type "t2v" --part $SLURM_ARRAY_TASK_ID

end=`date +%s`
runtime=$((end-start))

echo Runtime: $runtime