import os
import pandas as pd
import numpy as np

def get_data_from_filelist(filelist,selector=[]):
    """    
    Load csv files from filelist
    if selector is set, data will be filtered

    Parameters
    ----------
    filelist : array
        Filepath array
    selector : array
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
        dataset[filenum]=get_data(file,selector)
    return datadict

def get_data(filename,selector=[]):
    """
    Load csv file from filename
    if selector is set, data will be filtered

    Parameters
    ----------
    filename : string
        Filepath String
    selector : array
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
        df=df_filtering(df,selector)
    data=df.as_matrix().T
 
    return data

def df_filtering(df,selector=[]):
    """ 
    Select the dataframe column by looking up selector
    NaN value will be erased

    Parameters
    ----------
    df : dataframe
        Original Pandas DataFrame
    selector : array
        Selected column name array

    Returns
    -------
    result_df : dataframe
        Selected DataFrame
    """
    # selected index = column indexes having selector name
    selected_idx=[]
    # get column names from dataframe
    df_colname=df.columns.values

    # Find index of selector 
    for i in range(len(df_colname)):
        cur_col=df_colname[i]
        cur_col=cur_col.lower()
        # compare cur_col & col_name in selector
        for col_name in selector:
            # if dataframe column has one string of selector array
            if col_name in cur_col:
                selected_idx.append(i)
                break;
    result_df=df.iloc[:,selected_idx].dropna()
    return result_df

def get_filelist(dirname):
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










# step1 : Load data
def data_load(datadir):
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

    # 1-1 : load the filelist from data directory
    filelist=search(datadir)
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




