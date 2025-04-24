#!/bin/bash

#SBATCH --job-name=get_qwen_plot
#SBATCH --output=slurm/get_qwen_plot/new_distance_testing_qwen_%A_%a.out
#SBATCH --error=slurm/get_qwen_plot/new_distance_testing_qwen_%A_%a.err
#SBATCH --cpus-per-task=16
#SBATCH --mem=50Gb
#SBATCH --time=0:59:59
#SBATCH --array=0-119
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu-vram-12gb,gpu-vram-32gb,gpu-vram-48gb,gpu-vram-94gb
#SBATCH --mail-type=ALL
#SBATCH --mail-user=shashank.agnihotri@uni-mannheim.de

echo "Started at $(date)";
echo "Running job: $SLURM_JOB_NAME array id: $SLURM_ARRAY_TASK_ID using $SLURM_JOB_CPUS_PER_NODE cpus per node with given JID $SLURM_JOB_ID on queue $SLURM_JOB_PARTITION";

start=`date +%s`

coverages=(0.9 0.8 0.7 0.6 0.5 0.4 0.3 0.2 0.1 1.0)
components=(1 2 3 4 5 6 7 8 9 10 15 20)

coverage=${coverages[$SLURM_ARRAY_TASK_ID % 10]}
component=${components[$SLURM_ARRAY_TASK_ID / 12]}

#python -W ignore inference/get_distance_qwen.py --coverage $coverage --component $component
python -W ignore inference/new_distance_based_on_majority_bounds.py --coverage $coverage --component $component

end=`date +%s`
runtime=$((end-start))

echo Runtime: $runtime