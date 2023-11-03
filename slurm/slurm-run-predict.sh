#!/bin/bash
# Definining resource we want to allocate
#SBATCH --nodes=1
#SBATCH --ntasks=1

# 6 CPU cores per task to keep the parallel data feeding going. 
#SBATCH --cpus-per-task=6

# Allocate enough memory.
#SBATCH --mem=160G
#SBATCH -p gpu
###SBATCH -p gputest
# Time limit on Puhti's gpu partition is 3 days.
#SBATCH -t 08:00:00 ### update the time based on the size of the input files
###SBATCH -t 00:15:00
#SBATCH -J predict

# Allocate 4 GPUs on each node.
#SBATCH --gres=gpu:v100:1
#SBATCH --ntasks-per-node=1

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

#conda-containerize new --prefix conda-env env.yml
# export PATH="/projappl/project_2001426/BERT-based-entity-type-classifier/conda-env-c/bin:$PATH"
#python3 -m venv venv
source venv/bin/activate
#python -m pip install --upgrade pip
#python -m pip install nvidia-pyindex==1.0.5
#python -m pip install nvidia-tensorflow[horovod]==1.15.5

#change the paths to your local installation of openmpi

export PATH=${HOME}/openmpi/bin:$PATH
export LD_LIBRARY_PATH=${HOME}/openmpi/lib:$LD_LIBRARY_PATH


OUTPUT_DIR="output-biobert/multigpu/$SLURM_JOBID"
mkdir -p $OUTPUT_DIR



#check for all parameters
if [ "$#" -ne 9 ]; then
    echo "Usage: $0 model_dir data_dir max_seq_len batch_size task init_checkpoint labels_dir predictions_dir job_dir"
    exit 1
fi
#command example from BERT folder in projappl dir:
#sbatch slurm/slurm-run-predict-dis.sh models/biobert_v1.1_pubmed /scratch/project_2001426/stringdata/STRING-blocklists-v12/split-contexts 256 32 consensus /scratch/project_2001426/katerina/output-biobert/multigpu/19143192/model.ckpt-48828 data/biobert/other /scratch/project_2001426/stringdata/STRING-blocklists-v12/dis-predictions

BERT_DIR="$1"
DATASET_DIR="$2"
MAX_SEQ_LEN="$3"
BATCH_SIZE="$4"
TASK="$5"
INIT_CKPT="$6"
LABELS_DIR="$7"
PRED_DIR="$8" 
JOB_DIR="$9"

JOB_FILE=${JOB_DIR}/${SLURM_JOBID}

function on_exit {
   rm -rf ${JOB_FILE}
}
trap on_exit EXIT

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

python3 run_ner_consensus.py \
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

paste <(paste ${DATASET_DIR}"/test.tsv" ${OUTPUT_DIR}"/test_output_labels.txt") ${OUTPUT_DIR}"/test_results.tsv" | awk -F'\t' '{printf("%s\t%s\t%s\t%s\t%s\t%s\t%s\t\'\{\''che'\'': %s'\,' '\''dis'\'': %s'\,' '\''ggp'\'': %s'\,' '\''org'\'': %s'\,' '\''out'\'': %s'\}'\n",$1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12)}' > ${OUTPUT_DIR}"/output_with_probabilities_dict.tsv"; 
#remove everything up to last - 
mkdir -p $PRED_DIR
cp ${OUTPUT_DIR}"/output_with_probabilities_dict.tsv" ${PRED_DIR}"/output_with_probabilities_"$(basename ${DATASET_DIR##*-})".tsv"

echo -n 'result written in '"$PRED_DIR"$'\n'
seff $SLURM_JOBID
