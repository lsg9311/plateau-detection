import theano
import LSTM
import matplotlib.pyplot as plt
import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.regularizers import l1, l2, l1_l2
from keras.utils.vis_utils import plot_model
from keras.models import model_from_json

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
    
    reg={"bias":None,"kernel":l2(.01),"rec":l1_l2(.001), "activity":None}
    cell_num=env.get_config("model","dense_cell_num",type="int")
    
    # network structure
    model = Sequential()
    model=LSTM.generate_LSTM(model,env,reg,is_input=True,has_dropout=True)
    model=LSTM.generate_LSTM(model,env,reg,has_dropout=True)
    model=LSTM.generate_LSTM(model,env,reg,has_dropout=True)
    model=LSTM.generate_LSTM(model,env,reg,is_output=True,has_dropout=True)        
    model.add(Dense(cell_num,activation='relu', kernel_regularizer=reg["kernel"], bias_regularizer=reg["bias"],activity_regularizer=reg["activity"]))
    model.add(Dense(2,activation='softmax', kernel_regularizer=reg["kernel"], bias_regularizer=reg["bias"],activity_regularizer=reg["activity"]))

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
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

    epochs=env.get_config("model","epoch",type="int")
    batch_size=env.get_config("model","batch_size",type="int")

    for epoch in range(epochs):
        print("Epoch : "+  str(epoch)+" / "+str(epochs))
        for i in range(len(trainX)):
            X=trainX[i]; Y=trainY[i]
            model.fit(X, Y, epochs=1, batch_size=batch_size, verbose=2, shuffle=False)

    return model

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
    model.save_weights(h5_path)
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
    figure_range=4
    is_softmax=env.get_config("model","is_softmax",type="int")
    is_debug=env.get_config("system","debug",type="int")
    look_back=env.get_config ("data","look_back",type="int")

    testlist=env.file["test_file_list"]
    f1_result=0
    threshold=env.get_config("model","threshold",type="float")

    for test_idx in range(len(testlist)):
        if test_idx%figure_range==0:
            figure_num=figure_num+1
            plt.figure(figure_num,figsize=(12,5*figure_range))
        plt.subplot(4*100+10+((test_idx%figure_range)+1))
        testfile=testlist[test_idx]
        
        # for test
        if is_debug==1:
            testX,dataX,true=LSTM.make_test_file(testfile,env) 
            true=_decide_Y(true) if is_softmax==1 else _refine_Y(true,threshold)
            
        else:
            testX,dataX=LSTM.make_test_file(testfile,env)
        predY=model.predict(testX) 
        predY=_decide_Y(predY) if is_softmax==1 else _refine_Y(predY,threshold)

        if is_debug==1:
            f1_result=f1_result+f1_socre(true,predY)
            true=true*max(dataX[1])

        predY=predY*max(dataX[1])

        plt.plot(dataX[0][look_back:],dataX[1][look_back:],'b',label="ICP")
        if is_debug==1:
            plt.plot(dataX[0][look_back:],true.reshape(-1),'g',label="true")
        plt.plot(dataX[0][look_back:],predY.reshape(-1),'r',label="prediction")
        plt.grid()
        plt.legend()
    if is_debug==1:
        print("F1 RESULT : "+str(f1_result))
    plt.show()

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
