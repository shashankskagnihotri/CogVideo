#!/bin/bash

#SBATCH --job-name=guided_testing_gpt4_prompts
#SBATCH --output=slurm/gpt4_prompts_shield_guidance/testing_gpt4_nudity_%A_%a.out
#SBATCH --error=slurm/gpt4_prompts_shield_guidance/testing_gpt4_nudity_%A_%a.err
#SBATCH --cpus-per-task=32
#SBATCH --mem=300Gb
#SBATCH --time=119:59:59
#SBATCH --array=0-7%4
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu-vram-94gb
#SBATCH --mail-type=ALL
#SBATCH --mail-user=shashank.agnihotri@uni-mannheim.de

echo "Started at $(date)";
echo "Running job: $SLURM_JOB_NAME array id: $SLURM_ARRAY_TASK_ID using $SLURM_JOB_CPUS_PER_NODE cpus per node with given JID $SLURM_JOB_ID on queue $SLURM_JOB_PARTITION";

start=`date +%s`



if (($SLURM_ARRAY_TASK_ID == 0)); then
    python -W ignore inference/get_cogvideoX_hidden_states_with_shield_gemma2.py --testing_unsafe --use_shield_gemma2 --testing_gpt4_nudity --use_glm --use_abliterated --output_path "outputs/hidden_states_latent/testing_abliterated_glm_without_prompt_prefix_using_guidance"  --generate_type "t2v"    
elif (($SLURM_ARRAY_TASK_ID == 2)); then
    python -W ignore inference/get_cogvideoX_hidden_states_with_shield_gemma2.py --testing_safe --use_negative_prompt --use_shield_gemma2 --testing_gpt4_nudity --output_path "outputs/hidden_states_latent/testing_baseline_without_prompt_prefix_using_guidance"  --generate_type "t2v"
elif (($SLURM_ARRAY_TASK_ID == 5)); then
    python -W ignore inference/get_cogvideoX_hidden_states_with_shield_gemma2.py --testing_unsafe --use_shield_gemma2 --testing_gpt4_nudity --output_path "outputs/hidden_states_latent/testing_baseline_without_prompt_prefix_using_guidance"  --generate_type "t2v"
elif (($SLURM_ARRAY_TASK_ID == 4)); then
    python -W ignore inference/get_cogvideoX_hidden_states_with_shield_gemma2.py --testing_safe --use_shield_gemma2 --testing_gpt4_nudity --output_path "outputs/hidden_states_latent/testing_baseline_without_prompt_prefix_using_guidance"  --generate_type "t2v"
elif (($SLURM_ARRAY_TASK_ID == 3)); then
    python -W ignore inference/get_cogvideoX_hidden_states_with_shield_gemma2.py --testing_unsafe --use_negative_prompt --use_shield_gemma2 --testing_gpt4_nudity --use_glm --use_abliterated --output_path "outputs/hidden_states_latent/testing_abliterated_glm_without_prompt_prefix_using_guidance"  --generate_type "t2v"
elif (($SLURM_ARRAY_TASK_ID == 7)); then
    python -W ignore inference/get_cogvideoX_hidden_states_with_shield_gemma2.py --testing_safe --use_negative_prompt --use_shield_gemma2 --testing_gpt4_nudity --use_glm --use_abliterated --output_path "outputs/hidden_states_latent/testing_abliterated_glm_without_prompt_prefix_using_guidance"  --generate_type "t2v"
elif (($SLURM_ARRAY_TASK_ID == 6)); then
    python -W ignore inference/get_cogvideoX_hidden_states_with_shield_gemma2.py --testing_unsafe --use_negative_prompt --use_shield_gemma2 --testing_gpt4_nudity --output_path "outputs/hidden_states_latent/testing_baseline_without_prompt_prefix_using_guidance"  --generate_type "t2v"
elif (($SLURM_ARRAY_TASK_ID == 1)); then
    python -W ignore inference/get_cogvideoX_hidden_states_with_shield_gemma2.py --testing_safe --use_shield_gemma2 --testing_gpt4_nudity --use_glm --use_abliterated --output_path "outputs/hidden_states_latent/testing_abliterated_glm_without_prompt_prefix_using_guidance"  --generate_type "t2v"

fi


end=`date +%s`
runtime=$((end-start))

echo Runtime: $runtime



