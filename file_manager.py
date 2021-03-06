import pandas as pd


'''
File Manager CLass
Converts a dictionary to a pandas dataframe and saves it as a .csv file
'''


class FileManager():

    '''
    Specify the relative path to store the data
    '''
    def __init__(self, path):
        self.path = path

    '''
    Specify the filename and the dictionary that stores the data
    '''
    def write(self, filename, data):
        # expects data to be a dictionary where state name is key and list of state values as values 
        filepath = self.path + filename
        dataframe = pd.DataFrame.from_dict(data) 
        dataframe.to_csv(filepath, index=None)
        
        