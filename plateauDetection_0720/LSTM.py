import loader as ld
import preprocess as pp
import labeling as lb
import numpy as np
from env import Env

from keras.layers import GRU, Bidirectional, Dropout

######################
#       model        #
######################
def generate_LSTM(model,env,reg,is_output=False,is_input=False,has_dropout=False):
    cell_num=env.get_config("model","lstm_cell_num",type="int")

    # If current layer is input layer
    if is_input:
        look_back=env.get_config("data","look_back",type="int")

        model.add(Bidirectional(GRU(cell_num,return_sequences=not(is_output), 
        kernel_regularizer=reg["kernel"], bias_regularizer=reg["bias"], recurrent_regularizer=reg["rec"],activity_regularizer=reg["activity"]),
        input_shape=(look_back, 1)))
    else:
        model.add(Bidirectional(GRU(cell_num,return_sequences=not(is_output), 
        kernel_regularizer=reg["kernel"], bias_regularizer=reg["bias"], recurrent_regularizer=reg["rec"],activity_regularizer=reg["activity"])))

    if has_dropout:
        dropout=env.get_config("model","dropout",type="float")
        model.add(Dropout(dropout))

    return model


#####################
#       data        #
#####################
def make_LSTM_icp_dataset(datadict,env):
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
    Ydict : dict {filenum : array (batch_size,1(sigmoid) or 2(softmax))}
        dictionary of Y list which can be used for LSTM model
    '''
    Xdict=dict(); Ydict=dict()
    for filenum in range(len(datadict)):
        data=datadict[filenum]
        Xdict[filenum]=make_LSTM_X(data,env)
        Ydict[filenum]=make_LSTM_icpY(data,env)

    return Xdict, Ydict

def make_LSTM_icpY(data,env):
    # make Y
    look_back=env.get_config("data","look_back",type="int")
    feature_size=len(env.get_config("data","feature",type="list"))

    timelapse=data[0]
    feature=data[1]
    dataY=[]
    for i in range(len(timelapse)-look_back):
        dataY.append(feature[i+look_back]) # have to change
    Y=np.array(dataY)
    Y=Y.reshape(Y.shape[0],1)
    return Y

def make_LSTM_dataset(datadict,labeldict,env):
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
    Ydict : dict {filenum : array (batch_size,1(sigmoid) or 2(softmax))}
        dictionary of Y list which can be used for LSTM model
    '''
    Xdict=dict(); Ydict=dict()
    for filenum in range(len(datadict)):
        data=datadict[filenum]
        label=labeldict[filenum]
        Xdict[filenum]=make_LSTM_X(data,env)
        Ydict[filenum]=make_LSTM_Y(label,env)

    return Xdict, Ydict


def make_LSTM_X(data,env):
	# scaling
    data=pp.scaling_data(data)
    # make X
    look_back=env.get_config("data","look_back",type="int")
    feature_size=len(env.get_config("data","feature",type="list"))

    timelapse=data[0]
    feature=data[1]
    dataX=[]
    for i in range(len(timelapse)-look_back):
        dataX.append(feature[i:i+look_back]) # have to change
    X=np.array(dataX)
    X=X.reshape(X.shape[0],X.shape[1],1)
    return X

def make_LSTM_Y(label,env):
    # make XY
    look_back=env.get_config("data","look_back",type="int")

    total_time=len(label)
    dataY=[]
    is_softmax=env.get_config("model","is_softmax",type="int")
    for i in range(total_time-look_back):        
        if is_softmax==1:
            cur_Y=[0,0]
            cur_Y[label[i+look_back]]=1
            dataY.append(cur_Y)
        else:
            dataY.append(label[i+look_back])
    Y=np.array(dataY)
    if is_softmax==1:
        Y=Y.reshape(Y.shape[0],2)
    else:
        Y=Y.reshape(Y.shape[0],1)

    return Y

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
    feature=env.get_config("data","feature",type="list")
    time_slice=env.get_config("data","time_slice",type="int")
    is_debug=env.get_config("system","debug",type="int")

    # load data
    dataX=ld.get_data(filepath,feature)
    # simplify data
    dataX=pp.mean_simplify(dataX,len(feature),time_slice)
    
    #make LSTM data
    X = make_LSTM_X(dataX,env)
    if is_debug==1:
        label_path=env.get_config("path","label_path")
        labeldata=lb.load_label(label_path)
        label=lb.data_labeling(dataX,filepath,labeldata)
        Y = make_LSTM_Y(label,env)
        return X,dataX,Y
    else:
        return X,dataX
    
    

    