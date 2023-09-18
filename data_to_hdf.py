# -*- coding: utf-8 -*-
import os

import h5py
import numpy as np
from sortedcontainers import SortedList
from torch.utils.data import Dataset
from tqdm import tqdm
import librosa

# Name of the database in your directory
dataset_name = "small_dataset"

# Sampling Rates vary a lot in our dataset, so we normalize to 22050 (the defauuult in a lot of work)
sr = 22050
# 2 channels for stereo sound, I think they use this in the paper but Im really not sure
channels = 2

data = "train_data"

data_path = dataset_name + '/' + data + '/'
########################################
#  Loading the sound files
########################################

path = data_path + 'mixed/'
file_list = os.listdir(path)

bad_index = []
mixed_samples = []
for idx, file in enumerate(file_list):
    file_path = path + file
    y, curr_sr = librosa.load(file_path, sr=sr, mono=True, res_type='kaiser_fast', offset=0, duration=4)
    if len(y) == 88200: #Check if files have the right length (duration * sr)
        mixed_samples.append(y)
    else:
        print(idx, "wrong length audio file") #should just be a check in the end that doesnt get triggered if we have correct data
        bad_index.append(idx)
    

path = data_path + 'clean/'
file_list = os.listdir(path)

clean_samples = []
for idx, file in enumerate(file_list):
    file_path = path + file
    y, curr_sr = librosa.load(file_path, sr=sr, mono=True, res_type='kaiser_fast', offset=0, duration=4)
    if idx not in bad_index:
        clean_samples.append(y)

    
path = data_path + 'noise/'
file_list = os.listdir(path)

noise_samples = []
for idx, file in enumerate(file_list):
    file_path = path + file
    y, curr_sr = librosa.load(file_path, sr=sr, mono=True, res_type='kaiser_fast', offset=0, duration=4)
    if idx not in bad_index:
        noise_samples.append(y)
    
########################################
#  Create HDF and save the sounds there
########################################


partition = "small_" + data

hdf_dir = '\\hdf_files'
print(hdf_dir)

os.makedirs(os.getcwd() + hdf_dir, mode=0o777, exist_ok=True)
hdf_file = os.path.join(os.getcwd() + hdf_dir, partition + ".hdf5")


instruments = ["speech", "noise"]


with h5py.File(hdf_file, "w") as f:
    f.attrs["sr"] = sr
    f.attrs["channels"] = channels
    f.attrs["instruments"] = instruments
    
    for idx, mix_audio in enumerate(mixed_samples):
        
         clean_audio = clean_samples[idx]
         noise_audio = noise_samples[idx]
         # Get the separate samples in the right format
         source_audios = []
         source_audios.append(clean_audio)
         source_audios.append(noise_audio)
         source_audios = np.concatenate(source_audios, axis=0)
         # Create file
         grp = f.create_group(str(idx))
         grp.create_dataset("inputs", shape=mix_audio.shape, dtype=mix_audio.dtype, data=mix_audio)
         grp.create_dataset("targets", shape=source_audios.shape, dtype=source_audios.dtype, data=source_audios)
         grp.attrs["length"] = mix_audio.shape[0]
         grp.attrs["target_length"] = source_audios.shape[0]





























