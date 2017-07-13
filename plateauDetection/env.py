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
        self.train_file=[]
        self.test_file=[]
        self.feature=["datetime","icp"]

    def set_env(self):
        """
        set environment variable by using user input
        """
        menu=int_input("0: Set Feature")
        if menu==0:
            self._set_feature()

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


