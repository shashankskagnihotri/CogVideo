#!/bin/bash

#SBATCH --job-name=testing_glm_safety_cogvideox_academic_prompts
#SBATCH --output=slurm/testing_glm_abliterated_academic_prompts_again/testing_glm_%A_%a.out
#SBATCH --error=slurm/testing_glm_abliterated_academic_prompts_again/testing_glm_%A_%a.err
#SBATCH --cpus-per-task=32
#SBATCH --mem=550Gb
#SBATCH --time=7:59:59
#SBATCH --array=0-8
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu-vram-94gb
#SBATCH --mail-type=ALL
#SBATCH --mail-user=shashank.agnihotri@uni-mannheim.de

echo "Started at $(date)";
echo "Running job: $SLURM_JOB_NAME array id: $SLURM_ARRAY_TASK_ID using $SLURM_JOB_CPUS_PER_NODE cpus per node with given JID $SLURM_JOB_ID on queue $SLURM_JOB_PARTITION";

start=`date +%s`

if (($SLURM_ARRAY_TASK_ID == 0)); then
    python inference/safe_check_cli_demo.py --use_abliterated --use_glm --prefix_academic_prompts --testing_nudity --output_path "outputs/testing_ablitearted_glm4_with_academic_prompt_prefix_again" --generate_type "t2v"
elif (($SLURM_ARRAY_TASK_ID == 1)); then
    python inference/safe_check_cli_demo.py --use_abliterated --use_glm --prefix_academic_prompts --testing_inappropriate --output_path "outputs/testing_ablitearted_glm4_with_academic_prompt_prefix_again" --generate_type "t2v"
elif (($SLURM_ARRAY_TASK_ID == 2)); then
    python inference/safe_check_cli_demo.py --use_abliterated --use_glm --prefix_academic_prompts --testing_toxicity --output_path "outputs/testing_ablitearted_glm4_with_academic_prompt_prefix_again" --generate_type "t2v"
elif (($SLURM_ARRAY_TASK_ID == 3)); then
    python inference/safe_check_cli_demo.py --use_glm --prefix_academic_prompts --testing_nudity --output_path "outputs/testing_glm4_with_academic_prompt_prefix_again" --generate_type "t2v"
elif (($SLURM_ARRAY_TASK_ID == 4)); then
    python inference/safe_check_cli_demo.py --use_glm --prefix_academic_prompts --testing_inappropriate --output_path "outputs/testing_glm4_with_academic_prompt_prefix_again" --generate_type "t2v"
elif (($SLURM_ARRAY_TASK_ID == 5)); then
    python inference/safe_check_cli_demo.py --use_glm --prefix_academic_prompts --testing_toxicity --output_path "outputs/testing_glm4_with_academic_prompt_prefix_again" --generate_type "t2v"
elif (($SLURM_ARRAY_TASK_ID == 6)); then
    python inference/safe_check_cli_demo.py --prefix_academic_prompts --testing_nudity --output_path "outputs/testing_baseline_with_academic_prompt_prefix_again" --generate_type "t2v"
elif (($SLURM_ARRAY_TASK_ID == 7)); then
    python inference/safe_check_cli_demo.py --prefix_academic_prompts --testing_inappropriate --output_path "outputs/testing_baseline_with_academic_prompt_prefix_again" --generate_type "t2v"
elif (($SLURM_ARRAY_TASK_ID == 8)); then
    python inference/safe_check_cli_demo.py --prefix_academic_prompts --testing_toxicity --output_path "outputs/testing_baseline_with_academic_prompt_prefix_again" --generate_type "t2v"
fi


end=`date +%s`
runtime=$((end-start))

echo Runtime: $runtime
