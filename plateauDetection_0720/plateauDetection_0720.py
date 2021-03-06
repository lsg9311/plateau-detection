# custome module
import loader as ld
import preprocess as pp
import labeling as lb
import LSTM
import modelController as mc
import CNN
import numpy as np
import gc

from env import Env
from env import int_input
import matplotlib.pyplot as plt



if __name__ == '__main__':
    env=Env()
    env.load_config("./config.ini")

    print("Start Operation")
    print("Load Data")

    menu=int_input("0 : test pre_trained model / 1 : Pre-training / 2 : Generate post model / 3 : Post-training / 4 : Test Model")
    
    if menu==0:
        model=mc.load_model(env)
        imgfiles=ld._get_filelist("./data/train")
        testdict=ld.load_npimg(imgfiles)
        CNN.result_imgdict(model,testdict)
    if menu==1:
        imgfiles=ld._get_filelist("./data/train")
        traindict=ld.load_npimg(imgfiles)
    
        model=CNN.generate_cnn_autoencoder()
        trainX=CNN.make_cnn_X_all(traindict)

        model=CNN.train_encoder(model,trainX)
        mc.save_model(model,env)
    if menu==2:
        post_model=CNN.reorganize_model(env)
        post_model.summary()
        mc.save_model(post_model,env)
    if menu==3:
        model=mc.load_model(env)
        model.summary()
        imgfiles=ld._get_filelist("./data/train")
        traindict=ld.load_npimg(imgfiles)
        trainX=CNN.make_cnn_X_all(traindict)
        trainY=list()
        for i in range(len(trainX)):
            if (15 <= i <= 34) or (79 <= i <= 119) or (467 <= i <= 483):
                cur_time=1
            else:
                cur_time=0
            trainY.append(cur_time)
        trainY=np.array(trainY)
        print(trainY)
        model=CNN.train_model(model,trainX,trainY.reshape(trainY.shape[0],1))
        mc.save_model(model,env)
    if menu==4:
        model=mc.load_model(env)
        model.summary()
        imgfiles=ld._get_filelist("./data/test")
        testdict=ld.load_npimg(imgfiles)
        testX=CNN.test_cnn_X(testdict)
        pred=dict()
        for i in range(len(testX)):
            p=model.predict(testX[i])
            pred[i]=mc._refine_Y(p,threshold=0.5)
        np.save("prediction",pred)
    '''


    train_file,test_file,datadict=ld.data_load(env) # ./data = data file directory
    env.file["train_file_list"]=train_file
    env.file["test_file_list"]=test_file

    
    menu=int_input("0 : Load trained model / 1 : Generate new model / 2 : Get label")
    if menu==0:
        
        model=mc.load_model(env)
        print("Evaluation")
        mc.evaluate_model(model,env)
    elif (menu==1) or (menu==2) :
        if menu==2:
            env.config["system"]["label"]=1
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
            trainX,trainY=LSTM.make_LSTM_dataset(datadict,labeldict,env) #make_LSTM_icp_dataset(datadict,env)

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
        '''