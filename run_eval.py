#!/usr/bin/env python3

import os
import sys
import argparse
import subprocess

def external_eval_all_entities(pred_folder):
    cwd = os.path.dirname(os.path.realpath(__file__))
    command = "python3 evalsorel.py --entities Protein,Chemical,Complex,Family --relations Complex_formation data/farrokh_comparison/brat/devel/complex-formation-batch-02/" + pred_folder

    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE , cwd = cwd, shell=True)
    stdout, stderr = process.communicate()
    stdout = stdout.decode('utf-8').strip()
    stderr = stderr.decode('utf-8').strip()

    try:
        f_score = float(stdout.split (" F ")[-1].split("%")[0])
    except Exception as E:
        err_msg = "error in processing external evaluator output:\n"
        err_msg+= "cmd: " +  command + "\n"
        err_msg+= "stdout: " + stdout + "\n"
        err_msg+= "stderr: " + stderr + "\n"
        print(err_msg)
    return f_score , str({"OUT" : stdout , "ERR" : stderr})

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--writeback_devel_preds_folder", required=True, type=str)
    args = parser.parse_args()
    f_score, _ = external_eval_all_entities(args.writeback_devel_preds_folder)
    print (f_score)
