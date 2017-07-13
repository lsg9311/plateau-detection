import pandas as pd
import numpy as np

# load label file (=plateau time information)
def load_label(labelfile):
    pl_info=pd.read_csv(labelfile)
    pl_info=pl_info.iloc[:,[0,2,3]]
    pl_info=pl_info.as_matrix()
    return pl_info

# get plateau time information of one file
def get_pl_time(pl_info,file_name):
    pl_time=np.empty((0,2))
    for pl_row in pl_info:
        if pl_row[0] in file_name :
            pl_time=np.vstack((pl_time,pl_row[1:3]))
    
    return pl_time

# labelling one file
def pl_labeling(data,pl_time):
    label=[]
    time_lapse=data[0]
    for i in range(len(time_lapse)):
        time=time_lapse[i]
        is_pl=0
        for j in range(len(pl_time)):
            pl_start=pl_time[j][0]
            pl_end=pl_time[j][1]
            if pl_start<=time and pl_end>=time:
                is_pl=1
        label.append(is_pl)
    label=np.array(label)
    return label

def labelling(data,labelfile,filename):
    pl_info=load_label(labelfile)
    pl_time=get_pl_time(pl_info,filename)
    label=pl_labeling(data,pl_time)

    return label

# labelling all dataset
def data_labelling(dataset,labelfile,filenames):
    labelset=[]
    pl_info=load_label(labelfile)
    for i in range(len(dataset)):
        data=dataset[i]
        pl_time=get_pl_time(pl_info,filenames[i])
        labelset.append(pl_labeling(data,pl_time))
    return labelset