# custome module
import loader as ld
import preprocess as pp
import labeling as lb
import LSTM
import modelController as mc

from env import Env
from env import int_input

if __name__ == '__main__':
    env=Env()
    env.load_config("./config.ini")


    print("Start Operation")
    print("Load Data")
    train_file,test_file,datadict=ld.data_load(env) # ./data = data file directory
    env.file["train_file_list"]=train_file
    env.file["test_file_list"]=test_file
    
    menu=int_input("0 : Load trained model / 1 : Generate new model")
    if menu==0:
        
        model=mc.load_model(env)
        print("Evaluation")
        mc.evaluate_model(model,env)
    elif menu==1:
        if len(train_file)<=0:
            print("Train set is empty")
        else:
            print("Transform Data")
            feature=env.get_config("data","feature",type="list"); feature_size=len(feature)
            time_slice=env.get_config("data","time_slice",type="int")
            datadict=pp.transfrom_dataset(datadict,feature_size,time_slice)
    
            print("Labeling")
            label_path=env.get_config("path","label_path")
            labeldata=lb.load_label(label_path)
            labeldict=lb.dataset_labelling(datadict,env.file["train_file_list"],labeldata)
    
            print("Make XY")
            trainX,trainY=LSTM.make_LSTM_dataset(datadict,labeldict,env)

            
            print("Generate Model")
            model=mc.generate_model(env)
            print("Train Model")
            mc.train_model(model,env,trainX,trainY)
            print("Save Model")
            mc.save_model(model,env)
            print("Evaluation")
            mc.evaluate_model(model,env)
    else:
        print("wrong input")