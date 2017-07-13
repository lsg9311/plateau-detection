import os
import pandas as pd
import numpy as np
from env import int_input

def get_data_from_filelist(filelist,feature=[]):
    """    
    Load csv files from filelist
    if feature is set, data will be filtered

    Parameters
    ----------
    filelist : array
        Filepath array
    feature : array
        Selected column name array

    Returns
    -------
    datadict : dictionary {filenum : matrix(feature x timelapse)}
        Loaded data dictionary
    """
    # error check
    if len(filelist)<=0:
        print("You have to set the filelist")
        return

    # fill data dictionary
    datadict=dict()
    for filenum in len(filelist):
        file=filelist[filenum]
        dataset[filenum]=get_data(file,feature)
    return datadict

def get_data(filename,feature=[]):
    """
    Load csv file from filename
    if feature is set, data will be filtered

    Parameters
    ----------
    filename : string
        Filepath String
    feature : array
        Selected column name array

    Returns
    -------
    datamat : matrix (feature x timelapse)
        Loaded data matrix
    """
    # error check
    if filename=="":
        print("You have to set the file name")
        return

    # load csv file by using pandas
    df=pd.read_csv(filename)
    # when filter is set
    if len(col_names)>0:
        df=df_filtering(df,feature)
    data=df.as_matrix().T
 
    return data

def df_filtering(df,feature=[]):
    """ 
    Select the dataframe column by looking up feature
    NaN value will be erased

    Parameters
    ----------
    df : dataframe
        Original Pandas DataFrame
    feature : array
        Selected column name array

    Returns
    -------
    result_df : dataframe
        Selected DataFrame
    """
    # selected index = column indexes having feature name
    selected_idx=[]
    # get column names from dataframe
    df_colname=df.columns.values

    # Find index of feature 
    for i in range(len(df_colname)):
        cur_col=df_colname[i]
        cur_col=cur_col.lower()
        # compare cur_col & col_name in feature
        for col_name in feature:
            # if dataframe column has one string of feature array
            if col_name in cur_col:
                selected_idx.append(i)
                break;
    result_df=df.iloc[:,selected_idx].dropna()
    return result_df

def _get_filelist(dirname):
    """ 
    Get filepath list in input directory

    Parameters
    ----------
    dirname : string
        File Directory
    
    Returns
    -------
    filelist : array
        list of filepath string
    """
    # get all file path in directory
    filenames = os.listdir(dirname)
    filelist=[]

    # concatenate dirname + filename
    for filename in filenames:
        full_filename = os.path.join(dirname, filename)
        filelist.append(full_filename)
    
    return filelist

def _split_path(filelist):
    """
    get filepath list of training data & test data by using user input

    Parameters
    ----------
    filelist : array
        list of all filepath
    Returns
    -------
    trainpath : array
        list of train filepath
    testpath : array
        list of test filepath
    """
    train_size=5
    # train set = 0 ~ train_size
    # should be changed
    train_size=int_input("train_size?")
    trainpath=filelist[0:train_size]
    testpath=filelist[train_size:len(filelist)]

    return trainpath,testpath


def data_load(datadir):
    """
    Operate data loader
    1. load filelist
    2. split trainset
    3. set feature
    4. make data dictionary from train filelist

    Parameters
    ----------
    datadir : string
        directory of data
    Returns
    -------
    trainlist : array
        array of trainset filepath
    testlist : array
        array of testset filepath
    datadict : dictionary
        data dictionary of trainset
    """
    # 1-1 : load the filelist from data directory
    filelist=_get_filelist(datadir)
    # 1-2 : set train size
    trainlist,testlist=_get_trainpath(filelist)
    # 1-3 : set feature
    feature=["datetime","icp"]
    # 1-4 : maek dictionary from trainset
    datadict=get_data_from_filelist(trainlist,feature)


# step1 : Load data
def op(datadir):
    """Load raw csv files from data directory

    Parameters
    ----------
    datadir : string
        directory 
    Returns
    -------
    ts : array
        Signal time axis reference (seconds).
    filtered : array
        Filtered ECG signal.
    rpeaks : array
        R-peak location indices.
    templates_ts : array
        Templates time axis reference (seconds).
    templates : array
        Extracted heartbeat templates.
    heart_rate_ts : array
        Heart rate time axis reference (seconds).
    heart_rate : array
        Instantaneous heart rate (bpm).
    """

   
    # 1-2 : set train size
    '''
    print("File List ) ")
    print(filelist)'''
    train_size=5
    
    print("Set train size")
    train_size=int(input())
    
    # 1-3 : split train_file & test_file
    train_file=filelist[0:train_size]
    test_file=filelist[train_size:len(filelist)] # have to store

    # 1-4 : load train data
    col_names=["datetime","icp"]
    '''print("Select Column")
    col_names=input()
    col_names=col_names.split(",")'''

    dataset=dict()
    for i in range(len(train_file)):
        cur_file=train_file[i]
        # cur_file=os.path.join(datadir, cur_file)
        data=get_data(cur_file,col_names)
        dataset[i]=data
    return train_file,test_file,dataset




