# Contains functions for errors and convevrgence tests
import matplotlib.pyplot as plt
from mesh import Mesh
from solvers import Solver, BurgersSolver, AdvectionSolver
from problems import *
n_ghosts = 3 # does not affect when >= 2


def l1_error(exact, approx):
    n_cols = exact.shape[1] if np.ndim(exact) > 1 else np.size(exact)
    return 2./( n_cols-1. )*np.sum(np.abs(exact-approx))


def l2_error(exact, approx):
    n_cols = exact.shape[1] if np.ndim(exact) > 1 else np.size(exact)
    return np.sqrt(2./( n_cols-1. )*np.sum(np.abs(exact-approx)**2))


def linf_error(exact, approx):
    return np.max(np.abs(exact-approx))


def solve_for_steps(solver_class, step_list, params):
    solutions = []
    L = params['b'] - params['a']
    for step in step_list:
        print(f'{solver_class.__name__} with step {step:.2e}')
        N = int(L/step+1)
        mesh = Mesh(params['a'], params['b'], N, n_ghosts)
        solver = solver_class(mesh)
        solver.set_initial_conditions(mesh)
        solver.solve(params['T'])
        approx_solution = solver.solution[:, mesh.domain]
        solutions.append(approx_solution)
    return solutions


def collect_errors(step_list, exact_solution, approx_solutions, params):
    l1_errors_list, l2_errors_list, linf_errors_list = [], [], []
    L = params['b'] - params['a']
    for step, approx_solution in zip(step_list, approx_solutions):
        N = int(L/step+1)
        mesh = Mesh(params['a'], params['b'], N, n_ghosts)
        exact = exact_solution(mesh.nodes[mesh.domain], params['T'])

        l1_errors_list.append(l1_error(exact,approx_solution[0,:]))
        l2_errors_list.append(l2_error(exact,approx_solution[0,:]))
        linf_errors_list.append(linf_error(exact,approx_solution[0,:]))
    return l1_errors_list, l2_errors_list, linf_errors_list


def plot_comparison(step_list, exact_solution, approx_solutions, params):
    L = params['b'] - params['a']
    for step, approx_solution in zip(step_list, approx_solutions):
        N = int(L/step+1)
        mesh = Mesh(params['a'], params['b'], N, n_ghosts)
        fine_mesh = Mesh(params['a'], params['b'], 1001, n_ghosts)
        exact = exact_solution(fine_mesh.nodes[fine_mesh.domain], params['T'])

        plt.plot(fine_mesh.nodes[fine_mesh.domain], exact, 'r', label='exact')
        plt.plot(mesh.nodes[mesh.domain], approx_solution[0,:], 'b.--')
        plt.grid()
        plt.legend()
        plt.savefig(f'test_compareN{N}.pdf')
        plt.close()


def plot_pointwise_error(step_list, exact_solution, approx_solutions, params):
    L = params['b'] - params['a']
    for step, approx_solution in zip(step_list, approx_solutions):
        N = int(L/step+1)
        mesh = Mesh(params['a'], params['b'], N, n_ghosts)
        exact = exact_solution(mesh.nodes[mesh.domain], params['T'])
        pointwise_error = np.abs(exact - approx_solution[0, :])

        plt.plot(mesh.nodes[mesh.domain], pointwise_error, 'b')
        plt.grid()
        plt.savefig(f'test_errorN{N}.pdf')
        plt.close()


def plot_orders(step_list, error_lists=[], error_labels=[]):
    for error_list, error_label in zip(error_lists, error_labels):
        plt.loglog(step_list, error_list, '^-', label=error_label)
    max_error = max(list(map(max, error_lists)))
    norm_factor3 = 5*max_error/max(step_list)**3
    norm_factor5 = 0.5*max_error/max(step_list)**5
    plt.loglog(step_list, norm_factor3*np.array(step_list)**3, 'r--', label='Third')
    plt.loglog(step_list, norm_factor5*np.array(step_list)**5, 'g--', label='Fifth')
    plt.grid()
    plt.legend()
    plt.xlabel('dx')
    plt.savefig(f'test_orders.pdf')
    plt.close()


if __name__=='__main__':
    import os
    from pathlib import Path
    Path('test_analysis').mkdir(parents=True, exist_ok=True)
    os.chdir('test_analysis')

    step_list = 4e-2/2**np.arange(0,8) # Henrick2006
    step_list = 4e-2/2**np.arange(0,5)
    # step_list = 4e-2/2**np.arange(0,3)
    # step_list = [4e-2]
    T = 2.

    solutions = solve_for_steps(AdvectionSolver, step_list, {'a':-1, 'b':1, 'T':T})

    l1, l2, l3 = collect_errors(step_list, possible_es['Henrick2005'], solutions, {'a':-1, 'b':1, 'T':T})
    print(f"""L1 erros:    { [f'{x:.3e}' for x in l1 ]}
L2 errors:   { [f'{x:.3e}' for x in l2 ]}
Linf errors: { [f'{x:.3e}' for x in l3 ]}""")

    # plot_comparison(step_list, possible_es['Henrick2005'], solutions, {'a':-1, 'b':1, 'T':T})
    plot_pointwise_error(step_list, possible_es['Henrick2005'], solutions, {'a':-1, 'b':1, 'T':T})
    plot_orders(step_list, [l1,l2,l3], ['L1', 'L2', 'Linf'])

