# custome module
import loader as ld
import dataTransform as dt
import labelling as lb
import dataController as dc
import modelController as mc

from env import Env
from env import int_input

if __name__ == '__main__':
    env=Env()
    env.load_config("./config.ini")

    config=env.config_var

    print("Start Operation")
    print("Load Data")
    train_file,test_file,datadict=ld.data_load(config) # ./data = data file directory
    env.file["train_file"]=train_file
    env.file["test_file"]=test_file
    
    menu=int_input("0 : Load trained model / 1 : Generate new model")
    if menu==0:
        model=mc.load_model(config["path"]["model_load_path"],config["path"]["weight_load_path"])
    elif menu==1:
        print("Transform Data")
        datadict=dt.transfrom_dataset(datadict,len(config["data"]["feature"]),int(config["data"]["time_slice"]))
    
        print("Labeling")
        env.labeldata=lb.load_label(config["path"]["label_path"])
        labeldict=lb.dataset_labelling(datadict,env.file["train_file"],env.labeldata)
    
        print("Make XY")
        trainX,trainY=dc.make_LSTM_dataset(datadict,labeldict,int(config["data"]["look_back"]))

        print("Generate Model")
        model=mc.generate_model(int(config["model"]["cell_num"]),float(config["model"]["dropout"]),int(config["data"]["look_back"]))
        print("Train Model")
        mc.training_model(model,trainX,trainY,epochs=int(config["model"]["epoch"]))
        print("Save Model")
        mc.save_model(model,config["path"]["model_save_path"],config["path"]["weight_save_path"])
    else:
        print("wrong input")
    print("Evaluation")
    mc.evaluate(model,env)
