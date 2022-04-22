import shutil
import argparse
from collections import OrderedDict
import numpy as np
import re 

def get_all_files_with_extension (folder_address, file_extension, process_sub_folders = True):
    #IMPORTANT ... Extenstion should be like : "txt" , "a2"  ... WITHOUT DOT !
    all_files = []
    if process_sub_folders:
        for root, dirs, files in shutil.os.walk(folder_address):
            for file in files:
                if file.endswith("." + file_extension):
                    all_files.append(shutil.os.path.join(root, file))
        return (all_files)
    else:
        for file in shutil.os.listdir(folder_address):
            if file.endswith("." + file_extension): #".txt" ;
                all_files.append(folder_address + file)
        return (all_files)

def get_results_from_one_log_file(file_path):
    params = None
    score = None
    with open (file_path) as f:
        for line in f:
            if line.startswith("TEST-RESULT"):
                x = re.compile(r"^TEST-RESULT\s+init_checkpoint\s+(?P<init_checkpoint>\S+)\s+data_dir\s+(?P<data_dir>\S+)\s+max_seq_length\s+(?P<max_seq_length>\d+)\s+train_batch_size\s+(?P<train_batch_size>\d+)\s+learning_rate\s+(?P<learning_rate>\S+)\s+num_train_epochs\s+(?P<num_train_epochs>\d+)\s+accuracy\s+(?P<accuracy>\d+\.\d+)$")
                match = x.search(line)
                if match is not None:
                    params = match.groupdict()
                    score = match.groupdict()['accuracy']
                    # print(f'score: {score}')
                    # print(f'params: {params}')
    return score, params

def encode_params(params):
    encoded = ""
    # res = {}
    # for k,v in sorted(params, key=lambda x:x[0]):
    #     res[k] = str(v)

    encoded+= params['init_checkpoint'] \
              +"\tlearning_rate\t" + params['learning_rate'] + \
              "\tnum_train_epochs\t" + params['num_train_epochs'] + \
              "\ttrain_batch_size\t" + params['train_batch_size'] + \
              "\tmax_seq_length\t" + params['max_seq_length'] 

    return encoded

def float_decimal_points (num, n = 2):
    #pretty-print for floating point numbers with exactly N digits after .
    r = "{:." + str(n) + "f}"
    return r.format (num)

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("folder_address" , type=str, help="path to the folder containing logfiles.")
    parser.add_argument("output_filename" , type=str, help="tsv file name")
    args = parser.parse_args()

    results = {}
    for file_path in get_all_files_with_extension(args.folder_address , "out"):
        this_file_results , this_file_params = get_results_from_one_log_file(file_path)
        if (this_file_params is not None) and (this_file_results is not None):
            accuracy = this_file_results
            #auc_score = this_file_results['internal_evaluation_scores']['AUC_score']

            encoded_params = encode_params(this_file_params)
            if encoded_params in results:
                results[encoded_params].append ([accuracy])
            else:
                results[encoded_params]=[[accuracy]]


    max_accuracy = 0
    with open(args.output_filename , "wt") as f:
        for encoded_params, vals in sorted(results.items(), key=lambda x:x[0]):
            accuracies = [float(i[0]) for i in vals]
            #auc_scores = [i[1] for i in vals]

            # min = str(np.min(accuracies))
            # max = str(max(accuracies))
            avg = float_decimal_points(np.mean(accuracies), 2)
            std = float_decimal_points(np.std(accuracies), 4)
            # cnt = str(len(accuracies))

            #auc_avg = str(np.mean(auc_scores))
            #auc_std = str(np.std(auc_scores))
            #f.write(encoded_params + "\t" + cnt + "\t" + avg + "\t" + min + "\t" + max + "\t" + std + "\t" + auc_avg + "\t" + auc_std + "\n")
            # f.write(encoded_params + "\t" + cnt + "\t" + avg + "\t" + min + "\t" + max + "\t" + std + "\n")
            f.write(encoded_params + "\t" + str(accuracies) + "\t" + avg + "\t" + std  + "\n")
