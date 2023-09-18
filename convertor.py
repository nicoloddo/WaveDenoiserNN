import os
import random
import subprocess
from pydub import AudioSegment
import multiprocessing as mp
#read sound file

file_list = os.listdir("data1\\clips")

def find_sound(path,index_sound):
        # iterate through all file
  
    
        # Check whether file is in wave format or not
        file_path=file_list[index_sound]
        if file_list[index_sound].endswith(".mp3"):
         file_path = f"{path}\\{file_list[index_sound]}"
        return file_path

def stuff(sound,i):
    sound_wav = 'test{num}.wav'
    sound = AudioSegment.from_mp3(sound)
    sound.export(os.getcwd() + "\\data1\\wav_clips\\" + sound_wav.format(num=i), format="wav")

def start():
    sound_path = "data1\\clips"

    index_sound = 0

    ls = []
    for i in range(len(file_list)):
        if index_sound==8*4000:
            break
        sound = find_sound(sound_path,index_sound)
        ls.append((sound,index_sound))
        index_sound += 1

    pool = mp.Pool(8)
    results = pool.starmap(stuff, ls)