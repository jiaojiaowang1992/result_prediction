import glob
import os
import pickle
import numpy as np
import pandas as pd
loss_files_dir = "../parameter/val_results_BiLSTM"
params_dir = "../parameter/optimal_params_BiLSTM"
if not os.path.exists(params_dir):
    os.makedirs(params_dir)
datasets = ["bpic2012_accepted", "bpic2012_cancelled", "bpic2012_declined", "bpic2017_accepted", "bpic2017_cancelled",
            "bpic2017_declined", "sepsis_cases_1", "sepsis_cases_2", "sepsis_cases_3", "production", "traffic_fines",
            "hospital_billing"]
method_names = ["BiLSTM"]
cls_methods = ["BiLSTM"]
cls_params_names = ['lstmsize', 'dropout', 'n_layers', 'batch_size', 'optimizer', 'learning_rate', 'nb_epoch']

for dataset_name in datasets:
    for method_name in method_names:
        for cls_method in cls_methods:
            files = glob.glob("%s/%s" % (loss_files_dir, "loss_%s_%s_*.csv" % (dataset_name, method_name)))
            if len(files) < 1:
                continue
            dt_all = pd.DataFrame()
            for file in files:
                dt_all = pd.concat([dt_all, pd.read_csv(file, sep=";")], axis=0,ignore_index=True)

            dt_all = dt_all[dt_all["epoch"] >= 5]
            dt_all["params"] = dt_all["params"] + "_" + dt_all["epoch"].astype(str)

            cls_params_str = dt_all["params"][np.argmin(dt_all["val_loss"])]
            print(dataset_name)
            print(cls_params_str)
            best_params = {cls_params_names[i]: val for i, val in enumerate(cls_params_str.split("_"))}
            outfile = os.path.join(params_dir, "optimal_params_%s_BiLSTM_BiLSTM.pickle" % (dataset_name))
            with open(outfile, "wb") as fout:
                pickle.dump(best_params, fout)




