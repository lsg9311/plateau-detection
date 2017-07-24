import os
import pandas as pd
import numpy as np
from env import Env
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
    for filenum in range(len(filelist)):
        file=filelist[filenum]
        datadict[filenum]=get_data(file,feature)

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
    if len(feature)>0:
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

def data_load(env):
    """
    Operate data loader
    1. load filelist
    2. set feature
    3. make data dictionary from train filelist

    Parameters
    ----------
    env : Env
        model enviroment
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
    trainlist=_get_filelist(env.get_config("path","training_path"))
    testlist=_get_filelist(env.get_config("path","test_path"))
    # 1-2 : set feature
    feature=env.get_config("data","feature",type="list")
    # 1-3 : maek dictionary from trainset
    datadict=get_data_from_filelist(trainlist,feature)

    return trainlist, testlist, datadict
