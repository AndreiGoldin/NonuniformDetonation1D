# Contains everything to store date
import numpy as np
import subprocess
import matplotlib.pyplot as plt
import os
from pathlib import Path


class Writer:
    def __init__(self, name):
        self.name = name

    @classmethod
    def create_folders(cls):
        Path('test_data').mkdir(parents=True, exist_ok=True)
        Path('test_pics').mkdir(parents=True, exist_ok=True)
        Path('test_videos').mkdir(parents=True, exist_ok=True)

    @classmethod
    def write_solution(cls, array, file_name):
        np.savez(file_name+'.npz', u=array)

    @classmethod
    def plot_solution(cls, mesh, array, time, file_name):
        plt.figure(dpi=300)
        plt.plot(mesh.nodes, array[0,:], 'b')
        plt.title(f't = {time:.2f}')
        plt.grid()
        plt.tight_layout()
        plt.savefig('test_pics/'+file_name+'.png')
        plt.close()

    @classmethod
    def save_solution(cls, filename, array):
        np.savez(filename+'.npz', solution=array)

    @classmethod
    def make_video(cls, file_name=None):
        anim_status = subprocess.run(f'cd test_pics && ffmpeg -hide_banner -loglevel error -framerate 20 -start_number 0 -i image%03d.png -y -vf format=yuv420p test.mp4 && cd -',
                   shell=True, check=True)
        if anim_status.returncode == 0:
            subprocess.run('mkdir -p test_videos && cp test_pics/test.mp4 test_videos && rm test_pics/*', shell=True, check=True)
