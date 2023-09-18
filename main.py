import os.path

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as dsets
import numpy as np
from setuptools import glob

from WaveUNet.data.utils import load, corr_length_pad, write_wav
from model import CNNModule
import multiprocessing as mp
if __name__ == "__main__":
    mp.freeze_support()
    dataset = "small_dataset"
    newdataset = "small_dataset_v2"

    for set in ["train", "val", "test"]:
        for ver in ["mixed", "noise", "clean"]:
            tracks = glob.glob(os.path.join(dataset, set + "_data", ver, "*"))
            for track in tracks:
                audio, _ = load(track, sr=44100, mono=True)
                audio = corr_length_pad(audio, 176400)
                path = str(os.path.basename(track))
                if not os.path.exists(os.path.join(newdataset, set+ "_data", ver)):
                    os.makedirs(os.path.join(newdataset, set+ "_data", ver))
                write_wav(os.path.join(newdataset, set+ "_data", ver,path[0:len(path)-1]),audio,44100)


def setup_model():
    # test dataset to make sure that the training and such was working
    train_dataset = dsets.MNIST(root='./data1',
                                train=True,
                                transform=transforms.ToTensor(),
                                download=True)

    test_dataset = dsets.MNIST(root='./data1',
                               train=False,
                               transform=transforms.ToTensor(),
                               download=True)
    torch.Size([10000, 28, 28])
    batch_size = 250
    n_iters = 3000
    num_epochs = n_iters / (len(train_dataset) / batch_size)
    num_epochs = int(num_epochs)

    model = CNNModule(batch_size=batch_size)
    model.train(num_epochs=num_epochs, train_dataset=train_dataset, test_dataset=test_dataset)