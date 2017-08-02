import tensorflow
import LSTM
import matplotlib.pyplot as plt
import numpy as np
import sklearn.metrics
import labelmaker as lm

from keras.models import Sequential
from keras.layers import Dense, Dropout,BatchNormalization
from keras.regularizers import l1, l2, l1_l2
from keras.utils.vis_utils import plot_model
from keras.models import model_from_json
from keras.optimizers import adam

from sklearn.metrics import f1_score

def generate_model(env):
    '''
    generate the model by looking up config variable

    Parameters
    ----------
    cell_num : int, option
        # of cell in LSTM layer
    dropout : float, option
        dropout rate
    look_back : int, option
        length of X sequence
    layer_num : int, option
        # of LSTM layer
    Returns
    -------
    model : keras model
        generated model
    '''
    
    reg={"bias":None,"kernel":None,"rec":None, "activity":None}# l2(.01),l1_l2(.001)
    cell_num=env.get_config("model","dense_cell_num",type="int")
    look_back=env.get_config("data","look_back",type="int")
    is_softmax=env.get_config("model","is_softmax",type="int")

    # network structure
    model = Sequential()
    # model.add(BatchNormalization(input_shape=(look_back, 1)))
    model=LSTM.generate_LSTM(model,env,reg,is_input=True, has_dropout=True)
    model=LSTM.generate_LSTM(model,env,reg,has_dropout=True)
    model=LSTM.generate_LSTM(model,env,reg,has_dropout=True)
    model=LSTM.generate_LSTM(model,env,reg,is_output=True,has_dropout=True)
    #  model.add(Dense(cell_num,activation='sigmoid', kernel_regularizer=reg["kernel"], bias_regularizer=reg["bias"],activity_regularizer=reg["activity"]))
    if is_softmax==1:
        model.add(Dense(2,activation='softmax', kernel_regularizer=reg["kernel"], bias_regularizer=reg["bias"],activity_regularizer=reg["activity"]))
    else:
        model.add(Dense(1,activation='sigmoid', kernel_regularizer=reg["kernel"], bias_regularizer=reg["bias"],activity_regularizer=reg["activity"]))
    # model structure visualize
    model.summary()
    model_plot_path=env.get_config("path","model_plot_path")
    plot_model(model, to_file=model_plot_path)

    return model

def train_model(model,env,trainX,trainY):
    '''
    train the model by using trainX, trainY

    Parameters
    ----------
    model : keras model
        pre-trained model
    trainX, trainY : dictionary
        dictionary of X, Y
    epochs : int, option
        # of epoch
    Returns
    -------
    model : keras model
        trained model
    '''
    is_softmax=env.get_config("model","is_softmax",type="int")
    if is_softmax==1:
        model.compile(loss='categorical_crossentropy', optimizer='Nadam')#, metrics=['accuracy'])
    else:
        model.compile(loss='mean_squared_error', optimizer='Nadam')#, metrics=['accuracy'])
    epochs=env.get_config("model","epoch",type="int")
    batch_size=env.get_config("model","batch_size",type="int")

    for epoch in range(epochs):
        print("Epoch : "+  str(epoch+1)+" / "+str(epochs))
        order=_shuffle_order(len(trainX))
        for i in order:
            X=trainX[i]; Y=trainY[i]
            model.fit(X, Y, epochs=1, batch_size=900, verbose=2, shuffle=False) #len(X)

    return model

def _shuffle_order(size):
    order=np.arange(size)
    np.random.shuffle(order)
    return order

def save_model(model,env):
    '''
    save trained model into json & h5

    Parameters
    ----------
    model : keras model
        trained model
    json_path : string
        filepath for model
    h5_path : string
        filepath for weight
    '''
    json_path=env.get_config("path","model_save_path")
    h5_path=env.get_config("path","weight_save_path")
    # serialize model to JSON
    model_json = model.to_json()
    with open(json_path, "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights(h5_path,overwrite="True")
    print("Saved model to disk")

def load_model(env):
    '''
    load trained model from json & h5

    Parameters
    ----------
    json_path : string
        filepath for model
    h5_path : string
        filepath for weight

    Returns
    -------
    model : keras model
        trained model
    '''
    json_path=env.get_config("path","model_load_path")
    h5_path=env.get_config("path","weight_load_path")
    # load json and create model
    json_file = open(json_path, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(h5_path)
    print("Loaded model from disk")
    model=loaded_model

    return model

def evaluate_model(model,env):
    '''
    evaluate model by drawing graph of test data
    1. make test file
    2. prediction
    3. store graph
    4. show entire result

    Parameters
    ----------
    model : keras model
        trained model
    env : Env
        Environment of this system
    '''
    figure_num=0
    figure_range=5
    is_softmax=env.get_config("model","is_softmax",type="int")
    is_debug=env.get_config("system","debug",type="int")
    is_label=env.get_config("system","label",type="int")
    look_back=env.get_config ("data","look_back",type="int")
    future=env.get_config("data","future",type="int")


    testlist=env.file["test_file_list"]
    f1_result=0
    threshold=env.get_config("model","threshold",type="float")

    labeltime=np.empty((0,2))
    
    for test_idx in range(len(testlist)):
        if test_idx%figure_range==0:
            #if figure_num>0:
                #plt.savefig('./result/result'+str(figure_num)+'.png')
            figure_num=figure_num+1
            plt.figure(figure_num,figsize=(12,5*figure_range))
        plt.subplot(figure_range*100+10+((test_idx%figure_range)+1))
        testfile=testlist[test_idx]
        
        # for test
        if is_debug==1:
            testX,dataX,true=LSTM.make_test_file(testfile,env) 
            true=_decide_Y(true) if is_softmax==1 else _refine_Y(true,threshold)
            
        else:
            testX,dataX=LSTM.make_test_file(testfile,env)
        predY=model.predict(testX) 
        print (predY)
        predY=_decide_Y(predY) if is_softmax==1 else _refine_Y(predY,threshold)

        if is_label==1:
            curtime=lm.find_label_time(dataX[0][look_back+future:],predY)
            labeltime=np.vstack((labeltime,curtime))

        if is_debug==1:
            f1_result=f1_result+f1_score(true,predY)
            true=true*max(dataX[1])
        predY=predY*max(dataX[1])

        plt.plot(dataX[0][look_back+future:],dataX[1][look_back+future:],'b',label="ICP")
        if is_debug==1:
            plt.plot(dataX[0][look_back+future:],true.reshape(-1),'g',label="true")
        plt.plot(dataX[0][look_back+future:],predY.reshape(-1),'r',label="prediction")
        plt.grid()
        plt.legend()
    if is_debug==1:
        print("F1 RESULT : "+str(f1_result))
    if is_label==1:
        print(labeltime)
        # lm.time_to_file(labeltime,env)
    plt.show()
    #plt.savefig('./result/result'+str(figure_num)+'.png')

def _refine_Y(Y,threshold=0.5): # for sigmoid
    for i in range(len(Y)):
        cur_y=Y[i]
        Y[i]= 1 if cur_y>threshold else 0
    return Y

def _decide_Y(Y): # for softmax
    result=[]
    for cur_y in Y:
        if cur_y[0]>cur_y[1]:
            result.append(0)
        else:
            result.append(1)
    return np.array(result)
