# result_prediction
The initial event logs are originated from https://researchdata.4tu.nl/home/ and the preprocessed datasets for analyses are available in ../data.

Step1：This script trains the 16 parameter combinations for each approach. Execute this script after obtaing the random search parameters.
     python train_param_optim_AttBiLSTM.py <dataset> <method> <classifier> <params_str> <results_dir>

Step2: This script extracts the best parameters for a predictive model based on multiple training runs. 
      Execute this script after training several parameter settings using the script train_param_optim_AttBiLSTM.py.

     python extract_best_params_AttBiLSTM.py

Step3: This script evaluates the test result according to the best parameters from the previous training. 

   python prediction_final_AttBiLSTM.py <dataset> <method> <classifier> <params_str> <results_dir>