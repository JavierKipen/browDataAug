from datetime import datetime
import numpy as np
import pandas as pd
from tensorflow.keras.models import clone_model
import operator
import ipdb

def num_list_to_str(num_list):
    return '[{:s}]'.format(' '.join(['{:.4f}'.format(x) for x in num_list]))
def get_attribute_values_str(list_attr,base_class):
    str_out=""
    for i in range(len(list_attr)):
        str_out += list_attr[i] + ": "+ str(operator.attrgetter(list_attr[i])(base_class))  + "\n";
    return str_out

def dump_config(filename,modelTrainer,model,modelInfo, comment=""):
    with open(filename, 'w') as f:
        if comment != "":
            f.write("Comment:"+ comment + "\n")
        f.write("---Data augmentation config:---\n")
        f.write(get_attribute_values_str(["magnitude_rel_std","stretch_rel_std","apply_brownian_aug","noise_std","brow_std","fill_noise_std"],modelTrainer.da))
        f.write("---Data train config:---\n")
        f.write(get_attribute_values_str(["n_epochs_max","lr","batch_size","early_stopping_patience","brow_aug_use","validation_perc"],modelTrainer))
        f.write("---Model summary:---\n")
        if not (modelInfo is None):
            if modelInfo.model_type=="ResNet":
                f.write(get_attribute_values_str(["filter_size","block_layers","dense_1","dense_2","dropout_end","dropout_blocks","activation"],modelInfo))
            if modelInfo.model_type=="QuipuSkip":
                f.write(get_attribute_values_str(["filter_size","kernels_blocks","dense_1","dense_2","dropout_end","dropout_blocks","activation"],modelInfo))
        model.summary(print_fn=lambda x: f.write(x + '\n'))

def crossval_run_w_notes(modelTrainer,model_base,modelInfo,out_folder,comment="",title_file="",n_runs=20,tuning=True,all_data=False,save_model=False):

    str_time=datetime.today().strftime('%Y%m%d_%H-%M-%S');
    log_path=out_folder+str_time+"_"+title_file+"_log.txt";
    best_test_acc=0;
    dump_config(log_path,modelTrainer,model_base,modelInfo,comment=comment)
    for i in range(n_runs):
        modelTrainer.reset_history();
        model=clone_model(model_base);
        run_name=out_folder+str_time+"_"+title_file+"_run_"+str(i) + ".csv";
        if all_data:
            train_acc, test_acc, n_epoch=modelTrainer.train_w_all_data(model);
            init_data = [train_acc, test_acc, num_list_to_str(modelTrainer.train_losses), num_list_to_str(modelTrainer.train_aug_losses)]
            df_results = pd.DataFrame([init_data], columns=["train_acc","test_acc", "train_loss", "train_aug_loss"]);
        else:
            train_acc, valid_acc, test_acc, n_epoch=modelTrainer.train_es(model,tuning=tuning);
            init_data = [train_acc, valid_acc, test_acc, num_list_to_str(modelTrainer.train_losses), num_list_to_str(modelTrainer.valid_losses), num_list_to_str(modelTrainer.train_aug_losses)]
            df_results = pd.DataFrame([init_data], columns=["train_acc","valid_acc","test_acc", "train_loss", "valid_loss", "train_aug_loss"]);
        
        if save_model and df_results["test_acc"][0]>best_test_acc:
            model.save(out_folder+str_time+"_"+title_file+'_BestModel.keras')
            best_test_acc=df_results["test_acc"][0];
        df_results.to_csv(run_name, index=False)
