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
        self.path={}
        self.file={}
        self.data={}
        self.model={}

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
        self.path=config["path"]
        self.data=config["data"]
        self.model=config["model"]


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


