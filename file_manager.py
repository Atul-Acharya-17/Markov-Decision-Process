import pandas as pd

class FileManager():

    def __init__(self, path):
        self.path = path

    def write(self, filename, data):
        # expects data to be a dictionary where state name is key and list of state values as values 
        filepath = self.path + filename
        dataframe = pd.DataFrame.from_dict(data) 
        dataframe.to_csv(filepath, index=None)
        
        