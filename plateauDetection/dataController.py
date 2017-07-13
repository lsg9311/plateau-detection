import numpy as np
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

def make_LSTM_data(data,label,look_back):
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
    timelapse=data[0]
    feature=data[1]
    dataX,dataY=[],[]
    for i in range(len(timelapse)-look_back):
        dataX.append(feature[i:i+look_back])
        dataY.append(label[i+look_back])
    X,Y=np.array(dataX),np.array(dataY)
    X=X.reshape(X.shape[0],X.shape[1],1)
    Y=Y.reshape(Y.shape[0],1)

    return X,Y

def make_LSTM_dataset(datadict,labeldict,look_back):
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
        Xdict[filenum],Ydict[filenum]=make_LSTM_data(data,label,look_back)

    return Xdict, Ydict