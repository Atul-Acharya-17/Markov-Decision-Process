import pandas as pd


class FileManager():
    """
    File Manager CLass
    Converts a dictionary to a pandas dataframe and saves it as a .csv file

    Attributes
    ----------
    path : str
        Path to the folder name

    Methods
    ----------
    write(filename, data) : Writes the data to the file
    """

    def __init__(self, path):
        """
        Parameters
        ----------
        path : str
            Path to the folder name
        """

        self.path = path


    def write(self, filename, data):
        """
        Writes the data to the file

        Parameters
        ----------

        filename : str
            Name of the file to write
        data : dictionary
            dictionary to write
        """
        
        filepath = self.path + filename
        dataframe = pd.DataFrame.from_dict(data) 
        dataframe.to_csv(filepath, index=None)
        
        