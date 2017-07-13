import configparser

class Env(object):
    """
    Enviroment of this model
    Change occurs only in main function

    Instance
    ________
    train_file : array
        array of training set filepath
    test_file : array
        array of test set filepath
    feature : array
        array of feature string 
    """
    def __init__(self):
        self.path={"training_path":"","test_path":"","lable_path":"",
                   "model_load_path":"","weight_load_path":"",
                   "model_save_path":"","weight_save_path":""}
        self.file={"train_file":[],"test_file":[]}
        self.data={"feature" : []}
        self.model={"epoch" : 50, "batch_size" : 1000}

    def load_config(self,config_path):
        """
        load environment variable from config.ini

        Parameter
        _________
        config_path : string
            configuration file path 
        """
        config = configparser.ConfigParser()
        config.read(config_path)
        print(config.sections())

    def set_env(self):
        """
        set environment variable by using user input
        """
        menu=int_input("0: Set Feature / 1: Set time slice")
        if menu==0:
            self._set_feature()
        elif menu==1:
            self.time_slice=int_input("time slice?")

    def _set_feature(self):
        """
        set feature which will be selected from original data
        previous information will be deleted
        """
        self.feature=[]
        feature_name=""
        while feature_name!="q":
            print("Input Feature Name (Quit : q) ")
            feature_name=input()
            self.feature.append(feature_name)


def int_input(message=""):
    """
    transform user input into int format
    if user input is not valid(=input can't be interpreted as int), ask again

    Parameter
    _________
    message : string, option
        print explaining message 
    
    Return
    ______
    val : int
        int type of user input
    """
    if len(message)>0:
        print(message)
    while True:
        val=input()
        try:
            val=int(val)
        except:
            print("Wrong input")
            continue
        break
    return val


