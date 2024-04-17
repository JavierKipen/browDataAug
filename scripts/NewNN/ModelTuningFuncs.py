from datetime import datetime
import numpy as np
import pandas as pd
from tensorflow.keras.models import clone_model

def num_list_to_str(self,num_list):
    return '[{:s}]'.format(' '.join(['{:.3f}'.format(x) for x in num_list]))
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
        f.write(get_attribute_values_str(["stretch_std","magnitude_std","stretch_prob","noise_std","brow_std","opt_aug"],modelTrainer.da))
        f.write("---Data train config:---\n")
        f.write(get_attribute_values_str(["n_epochs_max","lr","batch_size","early_stopping_patience","brow_aug_use","validation_perc"],modelTrainer))
        f.write("---Model summary:---\n")
        if not (modelInfo is None):
            if modelInfo.model_type=="ResNet":
                f.write(get_attribute_values_str(["filter_size","block_layers","dense_1","dense_2","dropout_end","dropout_blocks","activation"],modelInfo))
            if modelInfo.model_type=="QuipuSkip":
                f.write(get_attribute_values_str(["filter_size","kernels_blocks","dense_1","dense_2","dropout_val","activation"],modelInfo))
        model.summary(print_fn=lambda x: f.write(x + '\n'))

def crossval_run_w_notes(modelTrainer,model_base,modelInfo,out_folder,comment="",title_file="",n_runs=20,tuning=True):

    str_time=datetime.today().strftime('%Y%m%d_%H-%M-%S');
    log_path=out_folder+str_time+"_"+title_file+"_log.txt";
    
    dump_config(log_path,modelTrainer,model,modelInfo,comment=comment)
    for i in range(n_runs):
        modelTrainer.reset();
        model=clone_model(model_base);
        train_acc, valid_acc, test_acc, n_epoch=modelTrainer.train_es(model,tuning=tuning);
        run_name=out_folder+str_time+"_"+title_file+"_run_"+str(i) + ".csv";
        init_data = [train_acc, valid_acc, test_acc, num_list_to_str(modelTrainer.train_losses), num_list_to_str(modelTrainer.valid_losses), num_list_to_str(modelTrainer.train_aug_losses)]
        df_results = pd.DataFrame(init_data, columns=["train_acc","valid_acc","test_acc", "train_loss", "valid_loss", "train_aug_loss"]);
        df_results.to_csv(run_name, index=False)