import os
import sys

path_common=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))+"/Common/";
sys.path.append(path_common) #Adds common path to import the python files

from ModelTrainer import ModelTrainer;
import operator
from datetime import datetime

def get_attribute_values_str(list_attr,base_class):
    str_out=""
    for i in range(len(list_attr)):
        str_out += list_attr[i] + ": "+ str(operator.attrgetter(list_attr[i])(base_class))  + "\n";
    return str_out

def dump_config(filename,modelTrainer,model,comment=""):
    with open('modelsummary.txt', 'w') as f:
        if comment != "":
            f.write("Comment:"+ comment + "\n")
        f.write("---Data augmentation config:---\n")
        f.write(get_attribute_values_str(["stretch_std","magnitude_std","stretch_prob","noise_std","brow_std","opt_aug"],modelTrainer.da))
        f.write("---Data train config:---\n")
        f.write(get_attribute_values_str(["n_epochs_max","lr","batch_size","early_stopping_patience","brow_aug_use"],modelTrainer.da))
        f.write("---Data load config:---\n")
        f.write(get_attribute_values_str(["min_perc_test","max_perc_test"],modelTrainer.dl))
        f.write("---Model summary:---\n")
        model.summary(print_fn=lambda x: f.write(x + '\n'))

def crossval_run_w_notes(filename,modelTrainer,model,comment="",title_file="",out_folder="../../results/ResNetTuning/",n_runs=20):
    out_folder_log=out_folder+"Logs/";
    out_folder_res=out_folder+"Results/";
    str_time=datetime.today().strftime('%Y%m%d_%H-%M-%S');
    modelTrainer.crossval_es(model,n_runs=n_runs,data_folder=out_folder_res+str_time+title_file+".csv");
    dump_config(out_folder_log+str_time+title_file+".txt",modelTrainer,model,comment=comment)

if __name__ == "__main__":
    mt=ModelTrainer();
    model=get_quipu_model();
    mt.n_epochs_max=3;
    mt.crossval_es(model,n_runs=2)
