from Configration import data_path
import pandas as pd
import numpy as np
import pickle
import h5py
import os

data_paths = ["E:\\Tasks\\09. Traffic Congestion Prediction\\DL-Traff-Graph-main\\METRLA", 
              "E:\\Tasks\\09. Traffic Congestion Prediction\\DL-Traff-Graph-main\PEMSBAY"]
file_h5_names = ["metr-la.h5", "pems-bay.h5"]
folder_names = ["METRLA", "PEMSBAY"]
data_names = ["METRLA", "PEMSBAY"]
first_key = ['df', 'speed']

def save_object(obj, filename, path):
    filename = os.path.join(path, filename)
    with open(filename + ".pkl", 'wb') as outp:
        pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)
    outp.close()

def load_object(filename, path):
    filename = os.path.join(path, filename)
    with open(filename + ".pkl", 'rb') as outp:
        loaded_object = pickle.load(outp, encoding='latin1')
    return loaded_object


def prepar_h5_data(data_name, data_folder, path, first_key, file_name):
    with h5py.File(os.path.join(path, file_name), 'r') as file:
        dataset = file[first_key]
        data = []
        for key in dataset.keys():
            data.append(np.array(dataset[key]))
        data[2] = np.array([int(x.decode('utf-8').strip("b'")) for x in data[2]])
        df_first_column = pd.DataFrame(data[1], columns=['time'])
        df_array = pd.DataFrame(data[-1], columns=[i for i in data[2]])
        df_final = pd.concat([df_first_column, df_array], axis=1)
        df_final['time'] = pd.to_datetime(df_final['time'], unit='ns')
        df_final.to_csv(os.path.join(data_path, os.path.join(data_folder, data_name+'.csv')), index=False)

# for ind, path in enumerate([data_paths]):
#     prepar_h5_data(data_names[ind], folder_names[ind], path, first_key[ind], file_h5_names[ind])


file_pkl_names = ["adj_mx", "adj_mx_bay"]
def prepare_pkl_data(file_name, data_folder, path):
    adj_mx = load_object(file_name, path)
    df = pd.DataFrame(list(adj_mx[1].items()), columns=['ID', 'Value'])
    df.to_csv(os.path.join(data_path, os.path.join(data_folder, 'ID Value data.csv')), index=False, encoding='utf-8-sig')
    adj_matrix = pd.DataFrame(adj_mx[2])
    adj_matrix.to_csv(os.path.join(data_path, os.path.join(data_folder, 'adj_matrix.csv')), index=False)

# for ind, path in enumerate(data_paths):
#     prepare_pkl_data(file_pkl_names[ind], folder_names[ind], path)
