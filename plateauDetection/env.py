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
        self.config_var={}
        self.file={}
        
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

        for section in config.sections():
            self.config_var[section]={}
            for option in config.options(section):
               self.config_var[section][option]=config.get(section,option)
        
        
        feature_list=self.config_var["data"]["feature"].split(",")
        self.config_var["data"]['feature']=feature_list
        


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


