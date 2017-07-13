# custome module
import loader as ld
# import labelling as lb
# import dataController as dc
# import modelController as mc


if __name__ == '__main__':
    dataset=[] # training data
    labelset=[] # training label (0: normal / 1: plateau)
    model=[]
    print("Start Operation")
    print("1. Load Data")
    train_file,test_file,datadict=ld.data_load("./data") # ./data = data file directory
    print(datadict)
    '''
    print("2. Transform Data")
    dataset=dc.transform_dataset(dataset)
    print("3. Labeling")
    labelset=lb.data_labelling(dataset,"./Plateau Info.csv",train_file)
    
    print("4. Make XY")
    trainX,trainY=dc.makeChunkedXY(dataset,labelset,feature=1,look_back=20) # 1=icp feature, 20=lookback(=timestep)
    print("5. Select Model")
    model=mc.make_model(trainX,trainY)
    print("6. Evaluation")
    mc.evaluate(model,test_file)
    '''