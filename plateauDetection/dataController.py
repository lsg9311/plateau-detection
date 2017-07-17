import numpy as np
import loader as ld
import labelling as lb
import dataTransform as dt
from sklearn.preprocessing import MinMaxScaler

def scaling_data(data):
    '''
    scaling range of one data into [0,1]
    based on datetime
    
    Parameters
    ----------
    data : matrix
        matrix of one data
    Returns
    -------
    result : matrix
        matrix of one data transformed into [0,1] range
    '''
    scaler = MinMaxScaler(feature_range=(0,1))
    result=scaler.fit_transform(data.T).T
   
    return result

def make_LSTM_X(data,config):
	# scaling
    data=scaling_data(data)
    # make XY
    look_back=int(config["data"]["look_back"])

    timelapse=data[0]
    feature=data[1]
    dataX=[]
    for i in range(len(timelapse)-look_back):
        dataX.append(feature[i:i+look_back])
    X=np.array(dataX)
    is_emb=int(config["model"]["embedding"])
    if is_emb==0:
        X=X.reshape(X.shape[0],X.shape[1],1)
    return X

def make_LSTM_data(data,label,config):
    '''
    scaling the data first
    then make X, Y to train or test the LSTM model
    X has the T~T+look_back-1 sequences
    X will predict the T+look_back value of Y
    
    Parameters
    ----------
    data : matrix
        matrix of one data
    label : array
        label array of one data
    look_back : int
        length of X sequence
    Returns
    -------
    X : 3d matrix (batch_size,timestep(=lookback),1(=feature))
        matrix of X which can be used for LSTM model
    Y : array (batch_size,1)
        list of Y which can be used for LSTM model
    '''
    # scaling
    data=scaling_data(data)
    # make XY
    look_back=int(config["data"]["look_back"])

    timelapse=data[0]
    feature=data[1]
    dataX,dataY=[],[]
    is_softmax=int(config["model"]["is_softmax"])
    for i in range(len(timelapse)-look_back):
        dataX.append(feature[i:i+look_back])
        
        if is_softmax==1:
            cur_Y=[0,0]
            cur_Y[label[i+look_back]]=1
            dataY.append(cur_Y)
        else:
            dataY.append(label[i+look_back])
    X,Y=np.array(dataX),np.array(dataY)
    is_emb=int(config["model"]["embedding"])
    if is_emb==0:
        X=X.reshape(X.shape[0],X.shape[1],1)
    if is_softmax==1:
        Y=Y.reshape(Y.shape[0],2)
    else:
        Y=Y.reshape(Y.shape[0],1)

    return X,Y

def make_LSTM_dataset(datadict,labeldict,config):
    '''
    make dict of X, Y to train or test the LSTM model
    X has the T~T+look_back-1 sequences of each file
    X will predict the T+look_back value of Y
    
    Parameters
    ----------
    datadict : dictionary
        dictionary of all data
    labeldict : dictionary
        label dictionary of all data
    look_back : int
        length of X sequence
    Returns
    -------
    Xdict : dict {filenum : 3d matrix (batch_size,timestep(=lookback),1(=feature))}
        dictionary of X matrix which can be used for LSTM model
    Ydict : dict {filenum : array (batch_size,1)}
        dictionary of Y list which can be used for LSTM model
    '''
    Xdict=dict(); Ydict=dict()
    for filenum in range(len(datadict)):
        data=datadict[filenum]
        label=labeldict[filenum]
        Xdict[filenum],Ydict[filenum]=make_LSTM_data(data,label,config)

    return Xdict, Ydict

def make_test_file(filepath,env):
    '''
    make test file into X for LSTM model 
    
    Parameters
    ----------
    filepath : string
        filepath of current test data
    env : Env
        Environment of system
    Returns
    -------
    X : 3d matrix (batch_size,timestep(=lookback),1(=feature))
        matrix of X which can be used for LSTM model
    dataX : matrix (feature * timestep)
        original data X
    '''
    config=env.config_var
    # load data
    dataX=ld.get_data(filepath,config["data"]["feature"])
    # simplify data
    dataX=dt.mean_simplify(dataX,len(config["data"]["feature"]),int(config["data"]["time_slice"]))
    # make LSTM data
    labeldata=lb.load_label(config["path"]["label_path"])
    label=lb.data_labeling(dataX,filepath,labeldata)
    X,Y=make_LSTM_data(dataX,label,config)

    return X,dataX,Y