import pandas as pd
import numpy as np

def find_label_time(timelapse,result):
    timeset=np.empty((0,2))
    progress=0
    for time_idx in range(len(result)):
        cur_label=result[time_idx]
        if cur_label==1:
            if progress==0:
                progress=1
                start_time=timelapse[time_idx]
        else:
            if progress==1:
                progress=0
                end_time=timelapse[time_idx-1]
                time_info=np.array([start_time,end_time])
                timeset=np.vstack((timeset,time_info))

    if progress==1:
        progress=0
        end_time=timelapse[time_idx-1]
        time_info=np.array([start_time,end_time])
        timeset=np.vstack((timeset,time_info))    

    return timeset

def time_to_file(timeset,env):
    timedf=pd.DataFrame(timeset,columns=["Start","End"])
    save_path=env.get_config("path","time_save_path")
    timedf.to_csv(save_path)