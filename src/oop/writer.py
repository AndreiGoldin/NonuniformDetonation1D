# Contains everything to store date
import numpy as np
import subprocess
import matplotlib.pyplot as plt
from matplotlib import figure
import os
from pathlib import Path
import physics

class Writer:
    def __init__(self, params):
        E, amp, wn = params['act_energy'],params['bump_amp'],params['bump_wn']
        self.data_dir =  f'halfwave_data'
        self.pics_dir =  f'E{E:.1f}amp{amp:.1f}wn{wn:.2f}_pics'
        self.video_dir = f'halfwave_videos'
        self.video_name = f'E{E:.1f}amp{amp:.1f}wn{wn:.2f}_video'
        self.solution_name = f'E{E:.1f}amp{amp:.1f}wn{wn:.2f}_solution'

    def create_folders(self):
        # Path('test_data').mkdir(parents=True, exist_ok=True)
        # Path('test_pics').mkdir(parents=True, exist_ok=True)
        # Path('test_videos').mkdir(parents=True, exist_ok=True)
        Path(self.data_dir).mkdir(parents=True, exist_ok=True)
        Path(self.pics_dir).mkdir(parents=True, exist_ok=True)
        Path(self.video_dir).mkdir(parents=True, exist_ok=True)

    def plot_solution(self, mesh, array, time, file_name, params, dpi=300):
        # density_grad = physics.density_gradient(array, params)
        # p = physics.pressure(array, params)
        _temperature = physics.temperature(array, params)
        reaction_rate = physics.reaction_rate_from_cons(array, params)
        density = array[0,:]
        fig = figure.Figure(dpi=dpi, tight_layout=True)
        ax = fig.subplots(1)
        ax.plot(mesh.nodes, reaction_rate,'r', label='Reaction rate')
        ax.plot(mesh.nodes, density, 'b', label='Density')
        ax.plot(mesh.nodes, _temperature, 'k', label='Temperature')
        # ax.plot(mesh.nodes, density_grad, 'b', label='Density gradient')
        # ax.plot(mesh.nodes, p, 'b', label='Pressure')
        ax.set_title(f't = {time:.2f}')
        ax.set_ylim(0., 18.)
        ax.grid()
        ax.legend(loc='upper right')
        fig.savefig(os.path.join(self.pics_dir,file_name)+'.png')
        plt.close()

    def save_solution(self, array):
        np.savez(os.path.join(self.data_dir,self.solution_name)+'.npz',
                solution=array)

    def make_video(self, file_name=None):
        anim_status = subprocess.run(f'cd {self.pics_dir} && ffmpeg -hide_banner '+
                f'-loglevel error -framerate 20 -start_number 0 -i image%03d.png '+
                f'-y -vf format=yuv420p {self.video_name}.mp4 && cd - 1> /dev/null',
                   shell=True, check=True)
        if anim_status.returncode == 0:
            subprocess.run(f'cp {self.pics_dir}/{self.video_name}.mp4 {self.video_dir} '+
                    f'&& rm -r {self.pics_dir}', shell=True, check=True)



# ffmpeg -hide_banner -loglevel error -framerate 20 -start_number 0 -i image%03d.png -y -vf format=yuv420p test.mp4
