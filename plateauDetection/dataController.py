import numpy as np
from sklearn.preprocessing import MinMaxScaler

# slice and scale one data
def transform_data(data,time_slice=120):
    result=mean_slice(data,time_slice)
    # result=scaling_data(result)

    return result

# transform all data
def transform_dataset(dataset):
    time_slice=120
    '''print("Set time slice")
    time_slice=int(input())'''

    for i in range(len(dataset)):
        cur_data=dataset[i]
        dataset[i]=transform_data(cur_data,time_slice)

    return dataset
# slice by using mean
def mean_slice(data,time_slice=120):
    #from data
    t_time=data[0];    t_icp=data[1];
    #sliced data
    time=list(); icp=list();

    mean_time=0;    mean_icp=0;  
    total_len=len(t_time);    cur_slice=0;
    for i in range(0,total_len):
        # save & initialize
        if cur_slice==time_slice:
            mean_time=mean_time/time_slice; mean_icp=mean_icp/time_slice;

            time.append(mean_time); icp.append(mean_icp); 
            
            mean_time=0;           mean_icp=0;            
            
            cur_slice=0
        # add
        mean_time=mean_time+t_time[i]
        mean_icp=mean_icp+t_icp[i]
    
        cur_slice=cur_slice+1
    
    mean_time=mean_time/cur_slice; mean_icp=mean_icp/cur_slice; 
    time.append(mean_time); icp.append(mean_icp); 
    
    data=np.array([time,icp])
    
    return data
'''
def mean_slice(data,time_slice=120):
    #from data
    t_time=data[0];    t_abp=data[1];    t_icp=data[2];
    
    #sliced data
    time=list();    abp=list(); icp=list(); cpp=list()   
    
    mean_time=0;    mean_abp=0;    mean_icp=0;  
    total_len=len(t_time);    cur_slice=0;
    
    for i in range(0,total_len):
        # save & initialize
        if cur_slice==time_slice:
            mean_time=mean_time/time_slice; mean_abp=mean_abp/time_slice; mean_icp=mean_icp/time_slice;

            time.append(mean_time); abp.append(mean_abp); icp.append(mean_icp); cpp.append(mean_abp-mean_icp);
            
            mean_time=0;            mean_abp=0;            mean_icp=0;            
            
            cur_slice=0
        # add
        mean_time=mean_time+t_time[i]
        mean_abp=mean_abp+t_abp[i]
        mean_icp=mean_icp+t_icp[i]
    
        cur_slice=cur_slice+1
    
    mean_time=mean_time/cur_slice; mean_abp=mean_abp/cur_slice;    mean_icp=mean_icp/cur_slice; 
    time.append(mean_time); abp.append(mean_abp); icp.append(mean_icp); cpp.append(mean_abp-mean_icp);   
    
    data=np.array([time,abp,icp,cpp])
    
    return data'''

def slice(data,time_slice=120):
    total_len=len(dataset)
    for i in range(total_len):
        data=dataset[i]
        dataset[i]=mean_slice(data,time_slice)
    return dataset

def scaling_data(data):
    scaler = MinMaxScaler(feature_range=(0,1))
    data=scaler.fit_transform(data.T).T
   
    return data

# transform one data into LSTM input & output
def makeXY(data,label,feature=1,look_back=20):
    if len(data)<=0:
        print("You have to load the data first")

    # feature selection
    selected_data=data[feature]
    dataX,dataY=[],[]
    # cut into LSTM XY
    for i in range(len(selected_data)-look_back):
        dataX.append(selected_data[i:i+look_back])
        dataY.append(label[i+look_back])

    X=np.array(dataX).reshape(-1,look_back)
    Y=np.array(dataY).reshape(-1,1)

    return X,Y

# transform all data into chunked LSTM X,Y
def makeChunkedXY(dataset,labelset,feature=1,look_back=20):
    if len(dataset)<=0:
        print("You have to load the data first")
    resultX=np.empty((0,look_back))
    resultY=np.empty((0,1))
    for i in range(len(dataset)):
        cur_data=dataset[i]
        cur_label=labelset[i]
        X,Y=makeXY(cur_data,cur_label,feature,look_back)
        resultX=np.vstack((resultX,X))
        resultY=np.vstack((resultY,Y))
    # reshape
    resultX=resultX.reshape(resultX.shape[0],resultX.shape[1],1)
    resultY=resultY.reshape(resultY.shape[0],1)

    return resultX,resultY