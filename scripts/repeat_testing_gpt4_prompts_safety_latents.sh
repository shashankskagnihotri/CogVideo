#!/bin/bash

#SBATCH --job-name=testing_gpt4_prompts
#SBATCH --output=slurm/gpt4_prompts/testing_gpt4_nudity_%A_%a.out
#SBATCH --error=slurm/gpt4_prompts/testing_gpt4_nudity_%A_%a.err
#SBATCH --cpus-per-task=32
#SBATCH --mem=300Gb
#SBATCH --time=23:59:59
#SBATCH --array=0-1
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu-vram-94gb
#SBATCH --mail-type=ALL
#SBATCH --mail-user=shashank.agnihotri@uni-mannheim.de

echo "Started at $(date)";
echo "Running job: $SLURM_JOB_NAME array id: $SLURM_ARRAY_TASK_ID using $SLURM_JOB_CPUS_PER_NODE cpus per node with given JID $SLURM_JOB_ID on queue $SLURM_JOB_PARTITION";

start=`date +%s`

if (($SLURM_ARRAY_TASK_ID == 0)); then
    python -W ignore inference/get_cogvideoX_hidden_states.py --testing_gpt4_nudity --testing_safe --output_path "outputs/hidden_states_latent/testing_baseline_without_prompt_prefix" --generate_type "t2v"
elif (($SLURM_ARRAY_TASK_ID == 1)); then
    python -W ignore inference/get_cogvideoX_hidden_states.py --use_glm --use_abliterated --testing_gpt4_nudity --testing_safe --output_path "outputs/hidden_states_latent/testing_abliterated_glm_without_prompt_prefix" --generate_type "t2v" 

fi


end=`date +%s`
runtime=$((end-start))

echo Runtime: $runtime



