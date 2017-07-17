import tensorflow
import dataController as dc
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Bidirectional
from keras.layers import LSTM,GRU, Embedding,Flatten
from keras.models import model_from_json
from keras.utils.vis_utils import plot_model
from keras import backend as K
from keras.regularizers import l1, l2, l1_l2
from env import Env
import pickle

def generate_model(cell_num,dropout,look_back,layer_num):
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
    model = Sequential()
    reg={"bias":l1(0.01),"kernel":None,"rec":None}
  
    model.add(Bidirectional(GRU(cell_num,return_sequences=True, kernel_regularizer=reg["kernel"], bias_regularizer=reg["bias"], recurrent_regularizer=reg["rec"]),input_shape=(look_back, 1)))
    model.add(Dropout(dropout))
    if layer_num>2:
        for i in range(2,layer_num):
            model=_generate_LSTM(model,cell_num,dropout,reg)
    model.add(Bidirectional(GRU(cell_num, kernel_regularizer=reg["kernel"], bias_regularizer=reg["bias"], recurrent_regularizer=reg["rec"])))
    model.add(Dropout(dropout))
    model.add(Dense(cell_num,activation='relu', kernel_regularizer=reg["kernel"], bias_regularizer=reg["bias"]))
    model.add(Dense(2,activation='softmax', kernel_regularizer=reg["kernel"], bias_regularizer=reg["bias"]))
    model.summary()
    plot_model(model, to_file='model.png')

    return model

def _generate_LSTM(model,cell_num,dropout,reg):
    model.add(Bidirectional(GRU(cell_num,return_sequences=True, kernel_regularizer=reg["kernel"], bias_regularizer=reg["bias"], recurrent_regularizer=reg["rec"])))
    model.add(Dropout(dropout))
    return model


def training_model(model,trainX,trainY,epochs=5):
    '''
    train the model by using trainX, trainY
    batch size will be the # of sequence in each trainX

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
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

    for epoch in range(epochs):
        print("Epoch : "+  str(epoch)+" / "+str(epochs))
        for i in range(len(trainX)):
            X=trainX[i]; Y=trainY[i]
            model.fit(X, Y, epochs=1, batch_size=len(X), verbose=2, shuffle=False)

    return model

def save_model(model,json_path,h5_path):
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
    # serialize model to JSON
    model_json = model.to_json()
    with open(json_path, "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights(h5_path)
    print("Saved model to disk")

def load_model(json_path,h5_path):
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

def _refine_Y(Y,threshold=0.5):
    '''
    refine float value into binary value (0 or 1)

    Parameters
    ----------
    Y : array
        prediction result (=float)
    threshold : float
        threshold of prediction
    Returns
    -------
    Y : array
        refined binary Y (0 or 1)
    '''
    # threshold=(max(Y)+min(Y))/2
    for i in range(len(Y)):
        cur_y=Y[i]
        Y[i]= 1 if cur_y>threshold else 0
    return Y

def _decide_Y(Y):
    result=[]
    for cur_y in Y:
        if cur_y[0]>cur_y[1]:
            result.append(0)
        else:
            result.append(1)
    return np.array(result)

def evaluate(model,env):
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
    figure_range=4
    is_softmax=int(env.config_var["model"]["is_softmax"])
    testlist=env.file["test_file"]
    for test_idx in range(len(testlist)):
        if test_idx%figure_range==0:
            figure_num=figure_num+1
            plt.figure(figure_num,figsize=(12,5*figure_range))
        plt.subplot(4*100+10+((test_idx%figure_range)+1))
        testfile=testlist[test_idx]
        testX,dataX=dc.make_test_file(testfile,env)
        predY=model.predict(testX)
        if is_softmax==1:
            predY=_decide_Y(predY)
        else:
            predY=_refine_Y(predY,float(env.config_var["model"]["threshold"]))
        predY=predY*max(dataX[1])

        look_back=int(env.config_var["data"]["look_back"])
        plt.plot(dataX[0][look_back:],dataX[1][look_back:],'b',label="ICP")
        plt.plot(dataX[0][look_back:],predY.reshape(-1),'r',label="prediction")
        plt.grid()
        plt.legend()
    plt.show()
