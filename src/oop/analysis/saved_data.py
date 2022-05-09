import numpy as np
import matplotlib.pyplot as plt
from mesh import Mesh


def plot_from_file(filename, mesh):
    solution = np.load(filename+'.npz')['solution']
    plt.figure(figsize=(10,10))
    plt.plot(mesh.nodes, solution[0,:], 'b')
    plt.grid()
    plt.tight_layout()
    plt.savefig(f'{filename}.pdf')
    plt.close()


def plot_comparison(filename1, filename2, mesh1, mesh2):
    solution1 = np.load(filename1+'.npz')['solution']
    solution2 = np.load(filename2+'.npz')['solution']
    plt.figure(figsize=(10,10))
    plt.plot(mesh1.nodes, solution1[0,:], 'b', label='Fine')
    plt.plot(mesh2.nodes, solution2[0,:], 'ro', label='Coarse',fillstyle='none')
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'test_data/compare.pdf')
    plt.close()

if __name__=='__main__':
    # params = {'a':-5, 'b':5, 'N':401, 'T':1.8, 'frame':'LFOR', 'Nt':400}
    params = {'a':-5, 'b':5, 'N':12801, 'T':1.8, 'frame':'LFOR', 'Nt':12800}
    mesh = Mesh(params['a'], params['b'], params['N'], 3)
    filename = f"test_data/Euler_Nx{params['N']}Nt12800"
    plot_from_file(filename,mesh)
    fine_mesh = Mesh(params['a'], params['b'], 12801, 3)
    coarse_mesh = Mesh(params['a'], params['b'], 401, 3)
    fine_solution = f"test_data/Euler_Nx12801Nt12800"
    coarse_solution = f"test_data/Euler_Nx401Nt12800"
    plot_comparison(fine_solution,coarse_solution,fine_mesh,coarse_mesh)

