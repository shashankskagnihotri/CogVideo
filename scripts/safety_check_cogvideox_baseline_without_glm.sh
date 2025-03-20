#!/bin/bash

#SBATCH --job-name=baseline_limited_safety_cogvideox
#SBATCH --output=slurm/baseline_limited/without_glm_%A_%a.out
#SBATCH --error=slurm/baseline_limited/without_glm_%A_%a.err
#SBATCH --cpus-per-task=16
#SBATCH --mem=550Gb
#SBATCH --time=7:59:59
#SBATCH --array=0-2
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu-vram-94gb
#SBATCH --mail-type=ALL
#SBATCH --mail-user=shashank.agnihotri@uni-mannheim.de

echo "Started at $(date)";
echo "Running job: $SLURM_JOB_NAME array id: $SLURM_ARRAY_TASK_ID using $SLURM_JOB_CPUS_PER_NODE cpus per node with given JID $SLURM_JOB_ID on queue $SLURM_JOB_PARTITION";

start=`date +%s`

if (($SLURM_ARRAY_TASK_ID == 0)); then
    python -W ignore inference/safe_check_cli_demo.py --model_path THUDM/CogVideoX1.5-5b --generate_type "t2v" --testing_nudity --output_path "outputs/testing_basemodel_without_glm"

elif (($SLURM_ARRAY_TASK_ID == 1)); then
    python -W ignore inference/safe_check_cli_demo.py --model_path THUDM/CogVideoX1.5-5b --generate_type "t2v" --testing_inappropriate --output_path "outputs/testing_basemodel_without_glm"

elif (($SLURM_ARRAY_TASK_ID == 2)); then
    python -W ignore inference/safe_check_cli_demo.py --model_path THUDM/CogVideoX1.5-5b --generate_type "t2v" --testing_toxicity --output_path "outputs/testing_basemodel_without_glm"
fi


end=`date +%s`
runtime=$((end-start))

echo Runtime: $runtime