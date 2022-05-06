# Contains everything to store date
import numpy as np
import subprocess
import matplotlib.pyplot as plt

class Writer:
    def __init__(self, name):
        self.name = name

    @classmethod
    def write_solution(cls, array, file_name):
        np.savez(file_name+'.npz', u=array)

    @classmethod
    def plot_solution(cls, mesh, array, file_name):
        plt.plot(mesh.nodes, array[0,:])
        plt.savefig('test_pics/'+file_name+'.png')
        plt.close()

    @classmethod
    def make_video(cls, file_name=None):
        anim_status = subprocess.run(f'cd test_pics && ffmpeg -hide_banner -loglevel error -framerate 20 -start_number 0 -i image%03d.png -y -vf format=yuv420p test.mp4 && cd -',
                   shell=True, check=True)
        if anim_status.returncode == 0:
            subprocess.run('mkdir -p test_videos && cp test_pics/test.mp4 test_videos', shell=True, check=True)
