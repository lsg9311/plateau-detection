import tensorflow
import numpy as np
import loader as ld
import labelling as lb
import dataController as dc
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Bidirectional
from keras.layers import LSTM,GRU
from keras.models import model_from_json

from sklearn.metrics import f1_score
import matplotlib.pyplot as plt

import pickle

def make_model(trainX,trainY):
    model=[]
    print("Generate New Model? No:0 / Yes:1")
    menu=int(input())
    if menu:
        print("1) Generate new model")
        model=generate_model()
        print("2) Training new model")
        epochs=5
        
        print("Epochs?")
        epochs=int(input())
        model=training_model(model,trainX,trainY,epochs=epochs)
        
        print("Save Trained Model? No:0 / Yes:1")
        menu2=int(input())
        if menu2:
            save_model(model)
    else:
        print("1) Load trained model")
        model = load_model()
    return model
def evaluate(model,test_file):
    for filenum in range(len(test_file)):
        filename=test_file[filenum]
        # load test_file
        data=ld.get_data(filename)
        # transform testX
        data=dc.transform_data(data)
        label=lb.labelling(data,"./Plateau Info.csv",filename)
        # evaluate
        model_f1_score(model, data, label)

def load_model():
    # load json and create model
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("model.h5")
    print("Loaded model from disk")
    model=loaded_model

    return model

def save_model(model):
    # serialize model to JSON
    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("model.h5")
    print("Saved model to disk")

def generate_model(cell_num=32,dropout=0.3,look_back=20):
    model = Sequential()
    model.add(Bidirectional(GRU(cell_num,return_sequences=True),input_shape=(look_back, 1)))
    model.add(Dropout(dropout))
    model.add(Bidirectional(GRU(cell_num,return_sequences=True)))
    model.add(Dropout(dropout))
    model.add(Bidirectional(GRU(cell_num)))
    model.add(Dropout(dropout))
    model.add(Dense(1,activation='sigmoid'))

    return model

def training_model(model,trainX,trainY,epochs=200,batch_size=1000):
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(trainX, trainY, epochs=epochs, batch_size=batch_size, verbose=1, shuffle=False)

    return model

def refine_Y(Y,threshold=0.5):
    threshold=(max(Y)+min(Y))/2
    for i in range(len(Y)):
        cur_y=Y[i]
        Y[i]= 1 if cur_y>threshold else 0
    return Y

def model_f1_score(model,data,label,threshold=0.5):
    look_back=20
    # transform
    testX,testY=dc.makeXY(data,label,feature=1,look_back=look_back)
    testX=np.reshape(testX,(testX.shape[0],testX.shape[1],1))

    true=testY
    pred=model.predict(testX)
    pred=refine_Y(pred,threshold)

    print(f1_score(true,pred))
    pred=pred*max(data[1])

    plt.plot(data[0][look_back:],data[1][look_back:],'b',label="ICP")
    plt.plot(data[0][look_back:],pred.reshape(-1),'r',label="prediction")

    plt.grid()
    plt.legend()    
    plt.show()
   