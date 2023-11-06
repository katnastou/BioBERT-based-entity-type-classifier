#!/bin/bash
#!/bin/bash
# Definining resource we want to allocate.
#SBATCH --nodes=1
#SBATCH --ntasks=40

# 5 CPU cores per task to keep the parallel data feeding going. 
#SBATCH --cpus-per-task=1

# Allocate enough memory.
#SBATCH --mem=162G
#SBATCH -p small
###SBATCH -p gputest

# Time limit on Puhti's small partition is 3 days. 72:00:00
#SBATCH -t 24:00:00
###SBATCH -t 00:30:00
#SBATCH -J genbloc

# Puhti project number
#SBATCH --account=Project_2001426

# Log file locations, %j corresponds to slurm job id. symlinks didn't work. Will add hard links to directory instead. Now it saves in projappl dir.
#SBATCH -o logs/%j.out
#SBATCH -e logs/%j.err

module purge
module load python-data

mkdir -p logs

TYPES="
ggp
che
org
dis
"

RUN_DIR="${1:-/scratch/project_2001426/stringdata/STRING-blocklists-v12}"

#generate local lists for everything
for type in $TYPES; do
    PRED_DIR="${RUN_DIR}/${type}-predictions"
    mkdir -p ${RUN_DIR}/local_blacklists-01-${type}
    for f in ${PRED_DIR}/output_with_probabilities_*; do echo $f; time python3 make_local_blacklist.py -m 1 -t 1.01 -s ${type} -T ${type} $f > ${RUN_DIR}/local_blacklists-01-${type}/$(basename $f); done


    if [ "$type" != "dis" ]; then
        python3 global_blacklist_candidates.py -m 2 -t 0.5 ${RUN_DIR}/local_blacklists-01-${type}/output_with_probabilities_* | sort -rn > ${RUN_DIR}/${type}_global_min_mentions_1_min_docs_2_threshold_0.5.tsv
        awk -F"\t" '{printf("%s\tt\n",$4)}' ${RUN_DIR}/${type}_global_min_mentions_1_min_docs_2_threshold_0.5.tsv > ${RUN_DIR}/global_blacklist_${type}_2col.tsv
    else
        #different threshold for diseases
        python3 global_blacklist_candidates.py -m 2 -t 0.85 ${RUN_DIR}/local_blacklists-01-${type}/output_with_probabilities_* | sort -rn > ${RUN_DIR}/${type}_global_min_mentions_1_min_docs_2_threshold_0.85.tsv
        awk -F"\t" '{printf("%s\tt\n",$4)}' ${RUN_DIR}/${type}_global_min_mentions_1_min_docs_2_threshold_0.85.tsv > ${RUN_DIR}/global_blacklist_${type}_2col.tsv
    fi

    if [[ "$type" == "che" || "$type" == "ggp" ]]; then
        python3 global_blacklist_candidates.py -m 2 -t 0.0 ${RUN_DIR}/local_blacklists-01-${type}/*.tsv | sort -rn > ${RUN_DIR}/global-non-${type}-probabilities-min-mentions-1-min-docs-2.tsv

        #local block/allowlist

        mkdir -p ${RUN_DIR}/local-blacklists-ratio-1000-${type}
        for f in ${RUN_DIR}/local_blacklists-01-${type}/*.tsv; do echo $f; python3 make_local_bw_lists.py --blacklist --ratio 1000 ${RUN_DIR}/global-non-${type}-probabilities-min-mentions-1-min-docs-2.tsv $f | sort -rn > ${RUN_DIR}/local-blacklists-ratio-1000-${type}/$(basename $f); done
        #allowlist/whitelist only for ggp
        if [ $type == "ggp" ]; then
            mkdir -p ${RUN_DIR}/local-whitelists-ratio-1e15-global-${type}-prob-0.3
            for f in ${RUN_DIR}/local_blacklists-01-${type}/*.tsv; do echo $f; python3 make_local_bw_lists.py --ratio 1000000000000000 --global-ggp-prob 0.3 ${RUN_DIR}/global-non-${type}-probabilities-min-mentions-1-min-docs-2.tsv $f | sort -rn > ${RUN_DIR}/local-whitelists-ratio-1e15-global-${type}-prob-0.3/$(basename $f); done

            cat ${RUN_DIR}/local-whitelists-ratio-1e15-global-${type}-prob-0.3/* > ${RUN_DIR}/local-whitelist-${type}-ratio-1e15-global-${type}-prob-0.3.tsv

            awk -F"\t" '{printf("%s\t%s\tf\n",$2, $3)}' ${RUN_DIR}/local-whitelist-${type}-ratio-1e15-global-${type}-prob-0.3.tsv > ${RUN_DIR}/local_whitelist_${type}_3col.tsv
        fi

        cat ${RUN_DIR}/local-blacklists-ratio-1000-${type}/* > ${RUN_DIR}/local-blacklists-ratio-1000-${type}.tsv
        awk -F"\t" '{printf("%s\t%s\tt\n",$2, $3)}' ${RUN_DIR}/local-blacklists-ratio-1000-${type}.tsv > ${RUN_DIR}/local_blacklist_${type}_3col.tsv
    fi
done

#Concatenate all auto local lists
cat ${RUN_DIR}/local_whitelist_ggp_3col.tsv ${RUN_DIR}/local_blacklist_ggp_3col.tsv ${RUN_DIR}/local_blacklist_che_3col.tsv | sort > ${RUN_DIR}/auto_local.tsv

#Concatenate all automated global lists into one list
cat ${RUN_DIR}/global_blacklist_che_2col.tsv ${RUN_DIR}/global_blacklist_ggp_2col.tsv ${RUN_DIR}/global_blacklist_dis_2col.tsv ${RUN_DIR}/global_blacklist_org_2col.tsv | sort -u > ${RUN_DIR}/auto_global.tsv
