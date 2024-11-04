import os
import pandas as pd


def import_multiple_csv(path, startwith=None, endwith=None):
    """
    Import multiple CSV files from a given directory path.
    Args:
        path (str): The directory path where the CSV files are located.
        startwith (str, optional): Filter files by starting with a specific string. Defaults to None.
        endwith (str, optional): Filter files by ending with a specific string. Defaults to None.
    Returns:
        pandas.DataFrame: A DataFrame containing the concatenated data from all the CSV files.
    """
    
    file_lst = os.listdir(path)
    if startwith:
        file_lst = [file for file in file_lst if file.startswith(startwith)]
    if endwith:
        file_lst = [file for file in file_lst if file.endswith(endwith)]
        
    df = pd.DataFrame()

    for file in file_lst:
        file_path = os.path.join(path, file)
        df = pd.concat([df, pd.read_csv(file_path)], ignore_index=True)

    return df
