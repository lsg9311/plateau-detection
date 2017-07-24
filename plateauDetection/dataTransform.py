import numpy as np

# simplify by using mean
def mean_simplify(data,feature_size,time_slice):
    '''
    Simplify the data to mean value within time_slice
    
    Parameters
    ----------
    data : matrix (feature*timelapse)
        matrix of one data
    feature_size : int
        size of selected feature
    time_slice : int
        max length of time slice to calculate the mean value
    Returns
    -------
    result : matrix (feature * sliced_timelapse)
        simplified data
    '''
    
    #simplified feature
    mean_features=[]
    for i in range(feature_size):
        mean_features.append([])

    # mean value of cur time slice
    mean_val= np.zeros(feature_size)

    total_len=len(data.T);    cur_slice=0;
    
    for cur_time in range(0,total_len):
        # save & initialize
        if cur_slice==time_slice:
            # save cur time slice result
            for feature_num in range(feature_size):
                mean_features[feature_num].append(mean_val[feature_num]/time_slice)

            # initialize
            mean_val= np.zeros(feature_size)
            cur_slice=0
        # add
        for feature_num in range(feature_size):
            mean_val[feature_num]=mean_val[feature_num]+data[feature_num][cur_time]
    
        cur_slice=cur_slice+1
    
    # save cur time slice result
    for feature_num in range(feature_size):
        mean_features[feature_num].append(mean_val[feature_num]/cur_slice)
    result=np.array(mean_features)
    
    return result

# transform all data set
def transfrom_dataset(datadict,feature_size,time_slice):
    '''
    Transform all data in datadict
    1. Mean Simplify by time slice
    
    Parameters
    ----------
    datadict : dict {filenum : matrix (feature*timelapse)}
        data dictionary
    feature_size : int
        size of selected feature
    time_slice : int
        max length of time slice to calculate the mean value
    Returns
    -------
    result : dict {filenum : matrix (feature*timelapse)}
        simplified data dictionary
    '''
    result=dict()
    for filenum in range(len(datadict)):
        cur_data=datadict[filenum]
        cur_data=mean_simplify(cur_data,feature_size,time_slice)
        result[filenum]=cur_data
    return result