#!/bin/bash
# Definining resource we want to allocate. We set 8 tasks, 4 tasks over 2 nodes as we have 4 GPUs per node.
#SBATCH --nodes=1
#SBATCH --ntasks=1

# 6 CPU cores per task to keep the parallel data feeding going. 
#SBATCH --cpus-per-task=6

# Allocate enough memory.
#SBATCH --mem=64G
#SBATCH -p gputest

# Time limit on Puhti's gpu partition is 3 days.
###SBATCH -t 48:00:00
#SBATCH -t 00:15:00

# Allocate 4 GPUs on each node.
#SBATCH --gres=gpu:v100:1
#SBATCH --ntasks-per-node=1

# Puhti project number
#SBATCH --account=Project_2001426

#Exclude nodes
#SBATCH --exclude=r04g05,r14g07,r15g08

# Log file locations, %j corresponds to slurm job id. symlinks didn't work. Will add hard links to directory instead. Now it saves in projappl dir.
#SBATCH -o logs/%j.out
#SBATCH -e logs/%j.err

# Clear all modules
module purge

#load tensorflow with horovod support
module load tensorflow/1.15-hvd

OUTPUT_DIR="output-biobert/multigpu/$SLURM_JOBID"
mkdir -p $OUTPUT_DIR

#uncomment to delete output!!!
#function on_exit {
#    rm -rf "$OUTPUT_DIR"
#    rm -f jobs/$SLURM_JOBID
#}
#trap on_exit EXIT

#check for all parameters
if [ "$#" -ne 7 ]; then
    echo "Usage: $0 model_dir data_dir max_seq_len batch_size task init_checkpoint labels_dir"
    exit 1
fi
#command example from BERT folder in projappl dir:
#sbatch slurm/slurm-run.sh models/biobert_large scratchdata/4-class-10K-w20 64 32 5e-6 4 consensus models/biobert_large/bert_model.ckpt

#models --> symlink to models dir in scratch
#scratchdata --> symlink to data dir in scratch

BERT_DIR=${1:-"models/biobert_large"}
DATASET_DIR=${2:-"scratchdata/4-class-10K-w20"}
MAX_SEQ_LEN="$3"
BATCH_SIZE="$4"
TASK=${5:-"consensus"}
INIT_CKPT=${6:-"models/biobert_large/bert_model.ckpt"}
LABELS_DIR="$7"
# #fix in case you want to use uncased models
# #start with this 
# if [[ $BERT_DIR =~ "uncased" ]]; then
#     cased="--do_lower_case"
# else
#     cased=""
# fi

###!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!#####
####PAY ATTENTION DO IT ONLY WHEN YOU COPY DATA
#uncomment to delete input data!
#function on_exit {
#    rm -rf "$DATASET_DIR"
#}
#trap on_exit EXIT
###!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!#####

cased="true"

if [ "$cased" = "true" ] ; then
    DO_LOWER_CASE=0
    CASING_DIR_PREFIX="cased"
    case_flag="--do_lower_case=False"
else
    DO_LOWER_CASE=1
    CASING_DIR_PREFIX="uncased"
    case_flag="--do_lower_case=True"
fi

#rm -rf "OUTPUT_DIR"
#mkdir -p "$OUTPUT_DIR"

#export NCCL_IB_HCA="^mlx5_1:1"

export NCCL_DEBUG=INFO

#export OMP_PROC_BIND=true
echo "START $SLURM_JOBID: $(date)"

srun python run_ner_consensus.py \
    --do_prepare=true \
    --do_train=false \
    --do_eval=true \
    --do_predict=true \
    --replace_span="[unused1]" \
    --task_name=$TASK \
    --init_checkpoint=$INIT_CKPT \
    --vocab_file=$BERT_DIR/vocab.txt \
    --bert_config_file=$BERT_DIR/bert_config.json \
    --data_dir=$DATASET_DIR \
    --output_dir=$OUTPUT_DIR \
    --eval_batch_size=$BATCH_SIZE \
    --predict_batch_size=$BATCH_SIZE \
    --max_seq_length=$MAX_SEQ_LEN \
    --use_fp16 \
    --use_xla \
    --horovod \
    --cased=$cased \
    --labels_dir=$LABELS_DIR


result=$(egrep '^INFO:tensorflow:  eval_accuracy' logs/${SLURM_JOB_ID}.err | perl -pe 's/.*accuracy \= (\d)\.(\d{2})(\d{2})\d+$/$2\.$3/')
echo -n 'TEST-RESULT'$'\t'
echo -n 'init_checkpoint'$'\t'"$INIT_CKPT"$'\t'
echo -n 'data_dir'$'\t'"$DATASET_DIR"$'\t'
echo -n 'max_seq_length'$'\t'"$MAX_SEQ_LENGTH"$'\t'
echo -n 'accuracy'$'\t'"$result"$'\n'

seff $SLURM_JOBID
