import pandas as pd
import pickle
import os


# Function to read the data file
def read_data(file_path, **kwargs):
    try:
        raw_data = pd.read_excel(file_path, **kwargs)
    except Exception as e:
        print(e)
    else:
        return raw_data


# Function to save python objects
def save_file(filename, model_data):
    """
        Function to save an object as pickle file
    """
    try:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, 'wb') as f:
            pickle.dump(model_data, f)
    except Exception as e:
        print(e)
    else:
        return


# Function to load models from pkl files
def load_file(filename):
    """
        Function to load a pickle object
    """
    try:
        # os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, 'rb') as f:
            loaded_data = pickle.load(f)
            # print(loaded_data)
    except Exception as e:
        print(e)
    else:
        return loaded_data

