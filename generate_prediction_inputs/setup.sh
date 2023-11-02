#!/bin/bash

# define functions
# Function to monitor job completion
monitor_job_completion() {
    local job_name=$1
    local USER=$2
    shift 2  # Shift the arguments to access the command
    command_to_execute="$@"  # The command to execute after removing the first two arguments
    job_id=$(squeue -h -o "%A" -u $USER -n $job_name)
    if [ -n "$job_id" ]; then
        echo "Found job ID for $job_name: $job_id"
        # Check if the job is still in the queue or running
        while squeue -u "$USER" | grep "$job_id" &> /dev/null; do
            sleep 60  # Adjust the sleep duration as needed
        done
        echo "Slurm job $job_name with ID $job_id has finished."
        # Execute the command
        eval "$command_to_execute"
    else
        echo "No active job found with the name $job_name."
    fi
}

#get tagger results with global dictionary
# you need to set up tagger first in case you haven't already
# comment the next lines if you have already set up tagger
git clone https://github.com/larsjuhljensen/tagger tagger
cd tagger
make tagcorpus #to skip swig
cd ..

#download the dictionary files and the corpus
wget https://zenodo.org/api/records/10008720/files/dictionary-files-tagger-STRINGv12.zip?download=1

wget https://a3s.fi/s1000/PubMed-input.tar.gz
wget https://a3s.fi/s1000/PMC-OA-input.tar.gz
unzip dictionary-files-tagger-STRINGv12.zip 
tar -xzvf PubMed-input.tar.gz 
tar -xzvf PMC-OA-input.tar.gz 

# Download the pre-trained and finetuned models
mkdir -p models
wget https://zenodo.org/api/records/10008720/files/bert-base-finetuned-large-set.tar.gz?download=1
wget http://nlp.dmis.korea.edu/projects/biobert-2020-checkpoints/biobert_v1.1_pubmed.tar.gz
tar -xzvf bert-base-finetuned-large-set.tar.gz -C models
tar -xzvf biobert_v1.1_pubmed.tar.gz -C models
rm *tar.gz *zip

#run tagger with curated blocklists only
sbatch slurm-tagger.sh dictionary-files-tagger-STRINGv12 . dictionary-files-tagger-STRINGv12/curated_global.tsv dictionary-files-tagger-STRINGv12/curated_local.tsv results tagger/tagcorpus

job_name="Jentag"  # Replace 'your_job_name' with the actual Slurm job name - check the slurm script for that
USER="katenast"

#generate matches in the correct format with identifiers
monitor_job_completion "$job_name" "$USER" sbatch slurm-create-matches.sh dictionary-files-tagger-STRINGv12/all_entities.tsv results/all_matches.tsv results/database_matches.tsv

#generate docs if not already generated
if [ ! -f "results/database_documents.tsv" ]; then
    monitor_job_completion "$job_name" "$USER" sbatch slurm-create-docs.sh dictionary-files-tagger-STRINGv12/excluded_documents.txt 'gzip -cd `ls -1 pmc/*.en.merged.filtered.tsv.gz` `ls -1r pubmed/*.tsv.gz` |' results/database_documents.tsv
fi

job_name="matches"
# Initial sort and split job
monitor_job_completion "$job_name" "$USER" sbatch sort-split.sh results/database_documents.tsv results/database_matches.tsv

# Monitoring and executing subsequent jobs
job_name="sortsp" 

monitor_job_completion "$job_name" "$USER" ./run-context-split.sh

#this is the job name of all the jobs submitted by the run-context-split.sh 
job_name="context"

monitor_job_completion "$job_name" "$USER" ../run_predict_batch_auto_all_types.sh

job_name="predict"
blocklists_dir="blocklists" # the dir should be the same as in the run_predict_batch_auto_all_types.sh script

monitor_job_completion "$job_name" "$USER" sbatch ../blocklist_generation/generate-blocklists.sh "$blocklists_dir"

