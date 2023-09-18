from re import S

import numpy as np
from pydub import AudioSegment
from pydub.playback import play
import os
import random

from mutagen.wave import WAVE
import soundfile as sf

## Extract file from .tar
# import tarfile
# import glob
# import os
# import shutil 

# def make_a_split(input_dir, split_num):
#     print("split {}".format(split_num))
#     tar_output = tarfile.open("split_" + str(split_num) + ".tar.gz", "w:gz")
#     for file_name in glob.glob(os.path.join(input_dir, "*")):
#         tar_output.add(file_name, os.path.basename(file_name))
#     tar_output.close()
#     shutil.rmtree(tmp_output_dir)
#     print("split {} done".format(split_num))

# count_per_split = 300000
# split = 10

# tmp_output_dir = "tmp/"

# tar = tarfile.open('en.tar.tar')

# for idx, tarinfo in enumerate(tar):
#     tar.extract(tarinfo, tmp_output_dir)
#     if idx > 0 and idx % count_per_split == 0:
#        make_a_split(tmp_output_dir, split)
#        split += 1
# tar.close()

# # to make a split
# if os.path.exists(tmp_output_dir):
#    make_a_split(tmp_output_dir, split)


# mixing the sounds
# we have 19227 sounds



def convert_to_PCM16(sound):
    data, samplerate = sf.read(sound)
    sf.write(sound, data, samplerate, subtype='PCM_16')


def find_noise(path):
    # iterate through all file
    file_list_noise = os.listdir(path)
    index = random.randrange(len(file_list_noise))
    # Check whether file is in wave format or not
    file_path = file_list_noise[index]

    if file_list_noise[index].endswith(".wav"):
        file_path = f"{path}\\{file_list_noise[index]}"
    return file_path


def find_sound(path, index_sound):
    file_list = os.listdir(path)
    # Check whether file is in wave format or not
    file_path = file_list[index_sound]
    if file_list[index_sound].endswith(".wav"):
        file_path = f"{path}\\{file_list[index_sound]}"
    return file_path


def sound_length(sound):
    # indicate sound as wav object
    audio = WAVE(sound)

    # contains all the metadata about the wavpack file
    audio_info = audio.info
    length = int(audio_info.length)
    return sound, length


def start():

    build_set("train_data", 18000)
    build_set("test_data", 6000)
    build_set("val_data", 6000)

def build_set(folder, amount):
    # Parent Directories
    parent_dir = os.getcwd()
    path = os.path.join(parent_dir, "dataset", folder)
    os.makedirs(path)
    for x in ["noise", "mixed", "clean"]:
        z = os.path.join(path, x)
        os.makedirs(z)

    for i in range(amount):

        # it uses for name of the output file

        # Read noise File
        num = random.randint(1, 10)
        noise_path = "data1\\UrbanSound8K\\audio\\fold" + str(num)

        noise = find_noise(noise_path)
        convert_to_PCM16(noise)
        # read sound file
        sound_path = "data1\\wav_clips"

        sound = find_sound(sound_path, i)
        convert_to_PCM16(sound)
        # function to find length of the sound

        # find the sound length
        sound_wav, length = sound_length(sound)
        # mix noise and sound
        if length <= 6:
            # mix sound with noise, starting at 0ms into sound)
            sound1 = AudioSegment.from_wav(sound_wav)
            sound2 = AudioSegment.from_wav(noise)[:6000]
            _, file = os.path.split(noise)
            filenr = file.split("-")[1]
            output = sound2.overlay(sound1)
            # save the result
            output.export(
                "dataset\\" + folder + "\\mixed\\mixed_sounds-{file_num}.wave".format(
                    file_num=str(i)+"-"+str(filenr)), format="wav")
            sound1.export(
                "dataset\\" + folder + "\\clean\\clean_sounds-{file_num}.wave".format(
                    file_num=str(i)), format="wav")
            sound2.export(
                "dataset\\" + folder + "\\noise\\noise_sounds-{file_num}.wave".format(
                    file_num=str(i)), format="wav")
