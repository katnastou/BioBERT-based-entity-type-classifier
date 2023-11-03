#!/bin/bash
# Definining resource we want to allocate. We set 8 tasks, 4 tasks over 2 nodes as we have 4 GPUs per node.
#SBATCH --nodes=1
#SBATCH --ntasks=4

# 6 CPU cores per task to keep the parallel data feeding going. 
#SBATCH --cpus-per-task=10

# Allocate enough memory.
#SBATCH --mem=200G
#SBATCH -p gpu
###SBATCH -p gputest
# Time limit on Puhti's gpu partition is 3 days.
#SBATCH -t 72:00:00
###SBATCH -t 00:15:00
#SBATCH -J 12.5M-4

# Allocate 4 GPUs on each node.
#SBATCH --gres=gpu:v100:4
#SBATCH --ntasks-per-node=4

#Exclude nodes where jobs are failing
#SBATCH --exclude=r04g05,r04g01,r14g07,r15g08,r01g04,r03g07,r16g01,r16g02,r04g06

# Puhti project number
#SBATCH --account=Project_2001426

# Log file locations, %j corresponds to slurm job id.
#SBATCH -o logs/%j.out
#SBATCH -e logs/%j.err

# Clear all modules
module purge
# module load tykky

# #conda-containerize new --prefix conda-env env.yml
# export PATH="/projappl/project_2001426/BERT-based-entity-type-classifier/conda-env-c/bin:$PATH"
#python3 -m venv venv
source venv/bin/activate
#python -m pip install --upgrade pip
#python -m pip install nvidia-pyindex==1.0.5
#python -m pip install nvidia-tensorflow[horovod]==1.15.5

export PATH=${HOME}/openmpi/bin:$PATH
export LD_LIBRARY_PATH=${HOME}/openmpi/lib:$LD_LIBRARY_PATH

OUTPUT_DIR="output-biobert/multigpu/$SLURM_JOBID"
mkdir -p $OUTPUT_DIR

# comment if you don't want to delete output
# function on_exit {
#    rm -rf "$OUTPUT_DIR"
#    rm -f jobs/$SLURM_JOBID
# }
# trap on_exit EXIT

#check for all parameters
if [ "$#" -ne 9 ]; then
    echo "Usage: $0 model_dir data_dir max_seq_len batch_size learning_rate epochs task init_checkpoint labels_dir"
    exit 1
fi
#command example from BERT folder in projappl dir:
#sbatch slurm/slurm-run-finetuning-big.sh models/biobert_v1.1_pubmed /scratch/project_2001426/data-may-2020/5-class-12.5M-w100-filtered-shuffled 256 32 2e-5 1 consensus models/biobert_v1.1_pubmed/model.ckpt-1000000 data/biobert/other


#models --> symlink to models dir in scratch
#scratchdata --> symlink to data dir in scratch


BERT_DIR="$1"
DATASET_DIR="$2"
MAX_SEQ_LENGTH="$3"
BATCH_SIZE="$4"
LEARNING_RATE="$5"
EPOCHS="$6"
TASK="$7"
INIT_CKPT="$8"
LABELS_DIR="$9"

## uncomment in case you want to use uncased models - it has to be in the model's name to work
# if [[ $BERT_DIR =~ "uncased" ]]; then
#     cased="--do_lower_case"
# else
#     cased=""
# fi

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


export NCCL_DEBUG=INFO
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
#https://horovod.readthedocs.io/en/latest/troubleshooting_include.html#running-out-of-memory
export NCCL_P2P_DISABLE=1

echo "START $SLURM_JOBID: $(date)"

mpiexec --bind-to socket -np 4 python3 run_ner_consensus.py \
    --do_prepare=true \
    --do_train=true \
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
    --max_seq_length=$MAX_SEQ_LENGTH \
    --learning_rate=$LEARNING_RATE \
    --num_train_epochs=$EPOCHS \
    --cased=$cased \
    --labels_dir=$LABELS_DIR \
    --use_xla \
    --use_fp16 \
    --horovod
    

result=$(egrep '^INFO:tensorflow:  eval_accuracy' logs/${SLURM_JOB_ID}.err | perl -pe 's/.*accuracy \= (\d)\.(\d{2})(\d{2})\d+$/$2\.$3/')
f_score=$(egrep '^INFO:tensorflow:  f1-score' logs/${SLURM_JOB_ID}.err | perl -pe 's/.*f1-score \= (\d)\.(\d{2})(\d{2})\d+$/$2\.$3/')
echo -n 'TEST-RESULT'$'\t'
echo -n 'init_checkpoint'$'\t'"$INIT_CKPT"$'\t'
echo -n 'data_dir'$'\t'"$DATASET_DIR"$'\t'
echo -n 'max_seq_length'$'\t'"$MAX_SEQ_LENGTH"$'\t'
echo -n 'train_batch_size'$'\t'"$BATCH_SIZE"$'\t'
echo -n 'learning_rate'$'\t'"$LEARNING_RATE"$'\t'
echo -n 'num_train_epochs'$'\t'"$EPOCHS"$'\t'
echo -n 'f-score'$'\t'"$f_score"$'\t'
echo -n 'accuracy'$'\t'"$result"$'\n'


seff $SLURM_JOBID

