import os
import sys

path_common=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))+"/Common/";
sys.path.append(path_common) #Adds common path to import the python files

from ModelTrainer import ModelTrainer;
from ModelFuncs import get_resnet_model,get_quipu_model
import operator
from datetime import datetime

def get_attribute_values_str(list_attr,base_class):
    str_out=""
    for i in range(len(list_attr)):
        str_out += list_attr[i] + ": "+ str(operator.attrgetter(list_attr[i])(base_class))  + "\n";
    return str_out

def dump_config(filename,modelTrainer,model,comment=""):
    with open(filename, 'w') as f:
        if comment != "":
            f.write("Comment:"+ comment + "\n")
        f.write("---Data augmentation config:---\n")
        f.write(get_attribute_values_str(["stretch_std","magnitude_std","stretch_prob","noise_std","brow_std","opt_aug"],modelTrainer.da))
        f.write("---Data train config:---\n")
        f.write(get_attribute_values_str(["n_epochs_max","lr","batch_size","early_stopping_patience","brow_aug_use","use_weights"],modelTrainer))
        f.write("---Data load config:---\n")
        f.write(get_attribute_values_str(["min_perc_test","max_perc_test"],modelTrainer.dl))
        f.write("---Model summary:---\n")
        model.summary(print_fn=lambda x: f.write(x + '\n'))

def crossval_run_w_notes(modelTrainer,model,comment="",title_file="",out_folder="../../results/ResNetTuning/",n_runs=20):
    out_folder_log=out_folder+"Logs/";
    out_folder_res=out_folder+"Results/";
    str_time=datetime.today().strftime('%Y%m%d_%H-%M-%S');
    modelTrainer.crossval_es(model,n_runs=n_runs,data_folder=out_folder_res+str_time+"_"+title_file+".csv");
    dump_config(out_folder_log+str_time+"_"+title_file+".txt",modelTrainer,model,comment=comment)

if __name__ == "__main__":
    import tensorflow as tf
    physical_devices = tf.config.list_physical_devices('GPU')
    tf.config.set_visible_devices(physical_devices[3], 'GPU')
    mt=ModelTrainer(lr=5e-4,batch_size=512,opt_aug=False,use_weights=True,n_epochs_max=2,track_losses=True);
    #mt.da.stretch_std=0.07;
    #model=get_quipu_model(n_dense_1=2048,n_dense_2=1024);
    model=get_resnet_model(filter_size=64, block_layers=[3,2,2], init_conv_kernel=3,init_pool_size=3,dense_1=2048,dense_2=1024,activation_fnc='relu')
    model.summary();
    #mt.train_es(model)
    crossval_run_w_notes(mt,model,comment="test",title_file="LR",n_runs=2)
    #dump_config("../../results/ResNetTuning/Logs/test.txt",mt,model,comment="Hola")