# Contains everything to store and plot data
import numpy as np
import subprocess
import matplotlib.pyplot as plt
from matplotlib import figure
import os
from pathlib import Path


class Writer:
    def __init__(self, name):
        self.name = name

    @classmethod
    def create_folders(cls, problem, callbacs):
        cls.data_folder = None
        cls.pics_folder = None
        cls.video_folder = None
        if callbacs["write final solution"] or callbacs["write speed"]:
            cls.data_folder = f'results/{problem}_data'
            Path(f'results/{problem}_data').mkdir(parents=True, exist_ok=True)
        if callbacs["plot final solution"] or callbacs["plot speed"]:
            cls.pics_folder = f'results/{problem}_pics'
            Path(f'results/{problem}_pics').mkdir(parents=True, exist_ok=True)
        if callbacs["write video"]:
            cls.video_folder = f'results/{problem}_videos'
            Path(f'results/{problem}_videos').mkdir(parents=True, exist_ok=True)

    @classmethod
    def write_solution(cls, nodes, array, filename):
        np.savez(os.path.join(cls.data_folder, filename+'.npz'),
                 nodes=nodes,
                 solution=array[:, 3:-3])

    @classmethod
    def write_speed(cls, t, D, filename):
        np.savez(os.path.join(cls.data_folder, filename+'.npz'), time=t, speed=D)

    @classmethod
    def plot_solution(cls, mesh, array, time, file_name, dpi=100):
        for nrow in range(array.shape[0]):
            fig = figure.Figure(dpi=dpi)
            ax = fig.subplots(1)
            # ax.plot(mesh.nodes, array[nrow,:], 'b')
            ax.plot(mesh.nodes[mesh.domain], array[nrow,3:-3], 'b')
            ax.set_title(f't = {time:.2f}')
            ax.grid()
            # size = fig.get_size_inches()*fig.dpi
            fig.savefig(os.path.join(cls.pics_folder, file_name+f'_var{nrow+1}.png'))
            plt.close()

    @classmethod
    def plot_speed(cls, time, speed, file_name, dpi=100):
        fig = figure.Figure(dpi=dpi)
        ax = fig.subplots(1)
        ax.plot(time, speed, 'b')
        ax.set_ylabel(f'D(t)')
        ax.set_xlabel(f't')
        ax.grid()
        fig.savefig(os.path.join(cls.pics_folder, file_name+'.pdf'), bbox_inches="tight")
        plt.close()

    @classmethod
    def write_video(cls, array, file_name='test'):
        for nrow in range(array.shape[0]):
            vid_name = f'{file_name}_var{nrow+1}'
            anim_command = f'cd {cls.pics_folder} && ffmpeg -hide_banner -loglevel error -framerate 20 -start_number 0 -i image%03d_var{nrow+1}.png -y -vf format=yuv420p {vid_name}.mp4 && cd - 1> /dev/null'
            anim_status = subprocess.run(anim_command, shell=True, check=True)
            if anim_status.returncode == 0:
                move_command =f'cp {cls.pics_folder}/{vid_name}.mp4 {cls.video_folder}'
                subprocess.run(move_command, shell=True, check=True)
        rm_command =f'rm {cls.pics_folder}/image* {cls.pics_folder}/*.mp4'
        subprocess.run(rm_command, shell=True, check=True)
