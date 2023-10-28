#!/bin/sh
MAX_JOBS=200
#process all folders and generate the output tsvs
mkdir -p delme
mkdir -p split-contexts
mkdir -p predictions

TYPES="
che
org
ggp
dis
"

for i in {000..299}; do
    for type in $TYPES; do
        while true; do
            jobs=$(ls predictions | wc -l)
            if [ $jobs -lt $MAX_JOBS ]; then break; fi
                echo "Too many jobs ($jobs), sleeping ..."
                sleep 60
        done
        echo "Submitting job ${type} ${i}"
        job_id=$(
        sbatch generate-contexts.sh \
            $type \
            $i \
            | perl -pe 's/Submitted batch job //'
        )
        echo "Submitted batch job $job_id"
        touch predictions/$job_id
        sleep 1
    done
done