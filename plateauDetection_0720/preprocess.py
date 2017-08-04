import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pytz
from datetime import datetime, timedelta

# scale one data into [0,1]
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

def fromOADate(v):
    return datetime(1899, 12, 30, 0, 0, 0, tzinfo=pytz.utc) + timedelta(days=v)

def get_group(data):
    timelapse=data[0]
    timelist=[]
    for i in range(len(timelapse)):
        curdt=fromOADate(timelapse[i])
        timelist.append(curdt)
    seclist=[]
    for i in range(len(timelist)):
        dt=timelist[i]
        seclist.append(dt.second)
    groups=[]
    cur_sec=0
    for i in range(len(seclist)):
        sec=seclist[i]
        if cur_sec!=sec:
            if cur_sec!=0:
                groups.append(np.array(cur_group))
            cur_sec=sec
            cur_group=[]
        cur_group.append(i)
    groups.append(np.array(cur_group))
    groups=np.array(groups)
    return groups

def group_by_sec(data,groups):
    new_data=[]
    for idxlist in groups:
        icp=0
        time=0
        for idx in idxlist:
            icp=icp+data[1][idx]
            time=time+data[0][idx]
        icp=icp/len(idxlist)
        time=time/len(idxlist)
        cur_data=[time,icp]
        new_data.append(np.array(cur_data))
    return np.array(new_data).T

def cut_by_hour(data,hour_limit=3):
    timelapse=data[0]
    progress_hour=0
    cur_hour=0
    
    total_data=[]
    start_idx=0
    for idx in range(len(timelapse)):
        curdt=fromOADate(timelapse[idx])
        dthour=curdt.hour
        if cur_hour!=dthour:
            cur_hour=dthour
            progress_hour=progress_hour+1
            if progress_hour>hour_limit:
                end_idx=idx
                sliced_data=data.T[start_idx:end_idx].T
                start_idx=idx
                
                total_data.append(sliced_data)
    end_idx=idx+1
    sliced_data=data.T[start_idx:end_idx].T
    total_data.append(sliced_data)
    return total_data

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
        groups=get_group(cur_data)
        cur_data=group_by_sec(cur_data,groups)
        # cur_data=mean_simplify(cur_data,feature_size,time_slice)
        result[filenum]=cur_data
    return result