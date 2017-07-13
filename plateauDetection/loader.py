import os
import pandas as pd
import numpy as np

# step1 : Load data
def data_load(datadir):
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

# get file list in directory
def search(dirname):
    filenames = os.listdir(dirname)
    filelist=[]
    for filename in filenames:
        full_filename = os.path.join(dirname, filename)
        filelist.append(full_filename)
    return filelist

# get data from one file
def get_data(file_name,col_names=["datetime","icp"]):
    if file_name=="":
        print("You have to set the file name")
        return
    df=pd.read_csv(file_name)
    df=df_filter(df,col_names)
    data=df.as_matrix().T
 
    return data

# col filter
def df_filter(df,col_names=["datetime","icp"]):
    selector=[]
    df_colname=df.columns.values
    
    for i in range(len(df_colname)):
        cur_col=df_colname[i]
        cur_col=cur_col.lower()
        for col_name in col_names:
            if col_name in cur_col:
                selector.append(i)
                break;
    df=df.iloc[:,selector].dropna()
    return df
