# custome module
import loader as ld
import dataTransform as dt
import labelling as lb
# import dataController as dc
# import modelController as mc

from env import Env
from env import int_input

if __name__ == '__main__':
    env=Env()
    env.load_config("./config.ini")

    config=env.config_var

    print("Start Operation")
    print("(1) Load Data")
    train_file,test_file,datadict=ld.data_load(config) # ./data = data file directory
    env.file["train_file"]=train_file
    env.file["test_file"]=test_file


    print("(2) Transform Data")
    datadict=dt.transfrom_dataset(datadict,len(config["data"]["feature"]),int(config["data"]["time_slice"]))
    
    print("(3) Labeling")
    env.labeldata=lb.load_label(env.config_var["path"]["label_path"])
    labeldict=lb.dataset_labelling(datadict,env.file["train_file"],env.labeldata)
    print(labeldict)

    '''
    
    print("4. Make XY")
    trainX,trainY=dc.makeChunkedXY(dataset,labelset,feature=1,look_back=20) # 1=icp feature, 20=lookback(=timestep)
    print("5. Select Model")
    model=mc.make_model(trainX,trainY)
    print("6. Evaluation")
    mc.evaluate(model,test_file)
    '''