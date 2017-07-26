import pandas as pd
import numpy as np
import re

# load label file (=plateau time information)
def load_label(labelfile):
    """
    load label file
    
    Parameters
    ----------
    labelfile : string
        labelfile path
    Returns
    -------
    labeldata : matrix(labelfile shape)
        matrix of label file
    """
    labeldf=pd.read_csv(labelfile)
    labeldata=labeldf.as_matrix()
    return labeldata

# get plateau time information of one file
def get_label_time(labeldata,file_name):
    """
    get label(=plateau) time of label file
    if filename in labeldata and filename of data is same, label time is added to list
    
    Parameters
    ----------
    labeldata : matrix
        matrix of entire label data
    file_name : string
        filename of data
    Returns
    -------
    labeltime : matrix(label * (start,end))
        start & end time of one file
    """
    # file_name parsing
    start_idx=file_name.find("\\")+1
    end_idx=file_name.find("_")
    #file_name=int(file_name[start_idx:end_idx])

    labeltime=np.empty((0,2))
    for label_row in labeldata:
        cur_row_name=label_row[0]
        # row_name parsing
        # cur_row_name = int(re.findall('\d+', cur_row_name)[0])
        #if cur_row_name == file_name :
        if cur_row_name in file_name:
            labeltime=np.vstack((labeltime,label_row[2:4]))
    
    return labeltime

# labelling one file
def data_labeling(data,file_name,labeldata):
    """
    make label of one file by looking up datetime of data
    
    Parameters
    ----------
    data : matrix
        matrix of original data
    file_name : string
        filename of data
    labeldata : matrix
        matrix of entire label data
    Returns
    -------
    label : array
        label of input data 0 : normal / 1 : plateau
    """

    # get time data
    labeltime=get_label_time(labeldata,file_name)
    
    label=[]
    time_lapse=data[0]
    print(labeltime)
    print(time_lapse[-1])
    for i in range(len(time_lapse)):
        time=time_lapse[i]
        is_pl=0
        # print(time)
        for j in range(len(labeltime)):
            pl_start=labeltime[j][0]
            pl_end=labeltime[j][1]
            if pl_start<=time and pl_end>=time:
                is_pl=1
        label.append(is_pl)
    label=np.array(label)
    return label

# labelling all dataset
def dataset_labelling(datadict,filelist,labeldata):
    """
    make label of all data by looking up datetime of data
    
    Parameters
    ----------
    datadict : dictionary
        dictionary of data
    filelist : array
        list of filepath
    labeldata : matrix
        matrix of entire label data
    Returns
    -------
    labeldict : dictionary
        dictionary of label
    """
    labeldict=dict()
    for i in range(len(datadict)):
        data=datadict[i]
        labeldict[i]=data_labeling(data,filelist[i],labeldata)
    return labeldict