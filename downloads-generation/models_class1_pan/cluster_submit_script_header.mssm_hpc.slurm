#!/bin/bash -l # not sure if login shell needed
#SBATCH --job-name= MHCf-{work_item_num} # Job name
#SBATCH --partition=gpu # queue name? how to request gpu nodes
#SBATCH --gpus 1
#SBATCH --ntasks=1# number of tasks
#SBATCH --cpus-per-task=18
#SBATCH --time=46:00:00 # hh:mm:ss
#SBATCH --mem=3000 # memory usage
#SBATCH --output={work_dir}/%j.stdout # output log (%j : JobID)
#SBATCH --error={work_dir}/STDERR-%J # error log
#

set -e
set -x

echo "Subsequent stderr output redirected to stdout" >&2
exec 2>&1

export TMPDIR=/local/JOBS/mhcflurry-{work_item_num} # same or not
export PATH=$HOME/miniconda3/envs/flurry/bin/python:$PATH # changed to my conda
export PYTHONUNBUFFERED=1
export KMP_SETTINGS=1

free -m

module load 2021 # module add or load
module load 2021_OSSC
module load CUDA/11.6.0
module load cuDNN/8.4.1.50-CUDA-11.6.0

module list

python -c 'import tensorflow as tf ; print("GPU AVAILABLE" if tf.test.is_gpu_available() else "GPU NOT AVAILABLE")'

env

cd {work_dir}

