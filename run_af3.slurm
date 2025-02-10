#!/bin/bash
#SBATCH --job-name=alphafold
#SBATCH --partition=hpcq
#SBATCH --nodelist=compute07
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=alphafold_%j.out
#SBATCH --error=alphafold_%j.err

# 激活Python虚拟环境
source /archive/liugu/alphafold-GPU/bin/activate

# 清除旧的环境变量（确保没有影响）
unset DB_DIR
unset XLA_PYTHON_CLIENT_MEM_FRACTION
unset CUDA_VISIBLE_DEVICES

# 设置环境变量
export DB_DIR=/archive/plato/task3/database
export JAX_FLASH_ATTENTION_IMPL=xla
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
export JAX_TRACEBACK_FILTERING=off

# 主程序只使用GPU 0
export CUDA_VISIBLE_DEVICES=0

# 运行AlphaFold
python /archive/liugu/alphafold3-GPU/run_alphafold.py \
  --json_path=/archive/liugu/input/processed/1491aa_5KIS.json \
  --output_dir=/archive/liugu/output-GPU \
  --model_dir=/archive/liugu/model \
  --norun_data_pipeline \
  --num_recycles=3 \
