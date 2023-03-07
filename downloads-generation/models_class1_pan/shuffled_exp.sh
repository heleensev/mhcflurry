#!/bin/bash
#
# Train pan-allele MHCflurry Class I models. Supports re-starting a failed run.
#
# Usage: GENERATE.sh <local|cluster> <fresh|continue-incomplete>
#
# cluster mode uses an HPC cluster (Mount Sinai chimera cluster, which uses lsf job
# scheduler). This would need to be modified for other sites.
#

#SBATCH --nodelist=dlc-groudon
#SBATCH --ntasks=10

source /home/dariomarzella/miniconda3/etc/profile.d/conda.sh
conda activate mhcflurry

set -e
set -x

# get name of the temporary directory working directory, physically on the compute-node
workdir="${TMPDIR}"

# get submit directory
# (every file/folder below this directory is copied to the compute node)
submitdir="${SLURM_SUBMIT_DIR}"

# change current directory to the location of the sbatch command
# ("submitdir" is somewhere in the home directory on the head node)
cd "${submitdir}"
# copy all files/folders in "submitdir" to "workdir"
# ("workdir" == temporary directory on the compute node)
cp -prf * ${workdir}
# change directory to the temporary directory on the compute-node
cd ${workdir}

DOWNLOAD_NAME=models_class1_pan_shuffled
SCRATCH_DIR=${TMPDIR-/tmp}/mhcflurry-downloads-generation
# SCRIPT_ABSOLUTE_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
SCRIPT_ABSOLUTE_PATH="$workdir"
# SCRIPT_DIR=$(dirname "$SCRIPT_ABSOLUTE_PATH")
SCRIPT_DIR="$workdir"

if [ "$1" != "cluster" ]
then
    GPUS=$(nvidia-smi -L 2> /dev/null | wc -l) || GPUS=0
    echo "Detected GPUS: $GPUS"

    PROCESSORS=$(getconf _NPROCESSORS_ONLN)
    echo "Detected processors: $PROCESSORS"

    if [ "$GPUS" -eq "0" ]; then
       NUM_JOBS=${NUM_JOBS-1}
    else
        NUM_JOBS=${NUM_JOBS-$GPUS}
    fi
    echo "Num jobs: $NUM_JOBS"
    PARALLELISM_ARGS+=" --num-jobs $NUM_JOBS --max-tasks-per-worker 1 --gpus $GPUS --max-workers-per-gpu 1"
else
    PARALLELISM_ARGS+=" --cluster-parallelism --cluster-max-retries 3 --cluster-submit-command sbatch --cluster-results-workdir $HOME/mhcflurry-scratch --cluster-script-prefix-path $SCRIPT_DIR/cluster_submit_script_header.mssm_hpc.slurm"
fi

mkdir -p "$SCRATCH_DIR/$DOWNLOAD_NAME"
if [ "$2" != "continue-incomplete" ]
then
    echo "Fresh run"
    rm -rf "$SCRATCH_DIR/$DOWNLOAD_NAME"
    mkdir "$SCRATCH_DIR/$DOWNLOAD_NAME"
else
    echo "Continuing incomplete run"
fi

# Send stdout and stderr to a logfile included with the archive.
LOG="$SCRATCH_DIR/$DOWNLOAD_NAME/LOG.$(date +%s).txt"
exec >  >(tee -ia "$LOG")
exec 2> >(tee -ia "$LOG" >&2)

# Log some environment info
echo "Invocation: $0 $@"
date

cd $SCRATCH_DIR/$DOWNLOAD_NAME

export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1

# cp $SCRIPT_DIR/additional_alleles.txt .

if [ "$2" != "continue-incomplete" ]
then
    cp $SCRIPT_DIR/generate_hyperparameters.py .
    python generate_hyperparameters.py > hyperparameters.yaml
fi
cp $SCRIPT_DIR/data/all_hla_pseudoseq.csv .
TRAINING_DATA="$(pwd)/all_hla_pseudoseq.csv"

for kind in combined
do
    CONTINUE_INCOMPLETE_ARGS=""
    if [ "$2" == "continue-incomplete" ]
    then
        echo "Will continue existing run: $kind"
        CONTINUE_INCOMPLETE_ARGS="--continue-incomplete"
    fi

    ALLELE_SEQUENCES="$(mhcflurry-downloads path allele_sequences)/allele_sequences.csv"
    HYPERPARAMETERS="hyperparameters.yaml"

    mhcflurry-class1-train-pan-allele-models \
        --data "$TRAINING_DATA" \
        --allele-sequences "$ALLELE_SEQUENCES" \
        --held-out-measurements-per-allele-fraction-and-max 0.25 100 \
        --num-folds 4 \
        --hyperparameters "$HYPERPARAMETERS" \
        --out-models-dir "$submitdir/output_shuffled/models.unselected.${kind}" \
        --worker-log-dir "$SCRATCH_DIR/$DOWNLOAD_NAME" \
        $PARALLELISM_ARGS $CONTINUE_INCOMPLETE_ARGS
done

echo "Done training. Beginning model selection."

for kind in combined
do
    MODELS_DIR="$submitdir/output_shuffled/models.unselected.${kind}"

    # Older method calibrated only particular alleles. We are now calibrating
    # all alleles, so this is commented out.
    #ALLELE_LIST=$(bzcat "$MODELS_DIR/train_data.csv.bz2" | cut -f 1 -d , | grep -v allele | uniq | sort | uniq)
    #ALLELE_LIST+=$(echo " " $(cat additional_alleles.txt | grep -v '#') )

    mhcflurry-class1-select-pan-allele-models \
        --data "$MODELS_DIR/train_data.csv.bz2" \
        --models-dir "$MODELS_DIR" \
        --out-models-dir models.${kind} \
        --min-models 2 \
        --max-models 8 \
        $PARALLELISM_ARGS
    cp "$MODELS_DIR/train_data.csv.bz2" "models.${kind}/train_data.csv.bz2"

done

cp $SCRIPT_ABSOLUTE_PATH .
bzip2 -f "$LOG"
for i in $(ls LOG-worker.*.txt) ; do bzip2 -f $i ; done
RESULT="$SCRATCH_DIR/${DOWNLOAD_NAME}.$(date +%Y%m%d).tar.bz2"
tar -cjf "$RESULT" *
echo "Created archive: $RESULT"

# Write out just the selected models
# Move unselected into a hidden dir so it is excluded in the glob (*).
mkdir .ignored
mv "$submitdir/output_shuffled/models.unselected.${kind}" .ignored/
RESULT="$SCRATCH_DIR/${DOWNLOAD_NAME}.selected.$(date +%Y%m%d).tar.bz2"
tar -cjf "$RESULT" *
mv .ignored/* . && rmdir .ignored
mv "$RESULT" "/data/cmbi/dario/"
echo "Created archive: $RESULT"
