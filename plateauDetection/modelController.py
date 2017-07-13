import tensorflow
import dataController as dc
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Bidirectional
from keras.layers import LSTM,GRU
from keras.models import model_from_json

from env import Env
import matplotlib.pyplot as plt

import pickle

def generate_model(cell_num=32,dropout=0.3,look_back=20):
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
    Returns
    -------
    model : keras model
        generated model
    '''
    model = Sequential()
    model.add(Bidirectional(GRU(cell_num,return_sequences=True),input_shape=(look_back, 1)))
    model.add(Dropout(dropout))
    model.add(Bidirectional(GRU(cell_num,return_sequences=True)))
    model.add(Dropout(dropout))
    model.add(Bidirectional(GRU(cell_num)))
    model.add(Dropout(dropout))
    model.add(Dense(1,activation='sigmoid'))

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
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

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

def prediction_result(model,testX,threshold):
    '''
    evaluate model by using one data(=testX)

    Parameters
    ----------
    model : keras model
        trained model
    testX : 3d matrix (batch_size,timestep(=lookback),1(=feature))
        matrix of testX
    Returns
    -------
    Y : array
        refined binary Y (0 or 1)
    '''

    Y=model.pred(testX)
    Y=_refine_Y(Y,threshold)
    return Y

def evaluate(model,env):
    testlist=env.file["test_file"]
    for testfile in testlist:
        testX=dc.make_test_file(testfile,env)
        Y=model.pred(testX)
        Y=_refine_Y(Y,env.config_var["model"]["threshold"])
   