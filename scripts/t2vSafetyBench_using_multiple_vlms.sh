#!/bin/bash

#SBATCH --job-name=t2vSafetyBench_using_multiple_vlms
#SBATCH --output=slurm/t2vSafetyBench_using_multiple_vlms/molmo_working_login_new_testing_VLMs_%A_%a.out
#SBATCH --error=slurm/t2vSafetyBench_using_multiple_vlms/molmo_working_login_new_testing_VLMs_%A_%a.err
#SBATCH --cpus-per-task=32
#SBATCH --mem=650Gb
#SBATCH --time=7:59:59
#SBATCH --array=1-1
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu-vram-94gb
#SBATCH --mail-type=ALL
#SBATCH --mail-user=shashank.agnihotri@uni-mannheim.de

echo "Started at $(date)";
echo "Running job: $SLURM_JOB_NAME array id: $SLURM_ARRAY_TASK_ID using $SLURM_JOB_CPUS_PER_NODE cpus per node with given JID $SLURM_JOB_ID on queue $SLURM_JOB_PARTITION";

start=`date +%s`

#models=("gpt-4o" "deepseek-r1" "paligemma2" "qwen2.5" "molmo")

#python -W ignore inference/multiple_VLMs_chat.py --model ${models[$SLURM_ARRAY_TASK_ID]}

if (( $SLURM_ARRAY_TASK_ID == 0 )); then
    python -W ignore inference/deepseek_r1_chat.py
elif (( $SLURM_ARRAY_TASK_ID == 1 )); then
    python -W ignore inference/molmo_chat.py
elif (( $SLURM_ARRAY_TASK_ID == 2 )); then
    python -W ignore inference/paligemma_2_chat.py
elif (( $SLURM_ARRAY_TASK_ID == 3 )); then
    python -W ignore inference/qwen_2.5_chat.py
elif (( $SLURM_ARRAY_TASK_ID == 4 )); then
    python -W ignore inference/gpt4_chat.py 
fi

end=`date +%s`
runtime=$((end-start))

echo Runtime: $runtime