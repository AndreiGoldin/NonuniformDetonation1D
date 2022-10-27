# Solver for hyperbolic problems in 1D
import time as timer
import numpy as np
from reader import Reader
from mesh import Mesh
from solvers import Solver, BurgersSolver, AdvectionSolver
from safor_solvers import SAFORSolver
from writer import Writer

def simulate(config_file='config.yaml'):
    """ Main function to read configuration, construct a solver, run simulation, and save results. """
    yaml_parser = Reader('yaml')
    param_dict, callbacks, param_tag, save_tag = yaml_parser.get_input(config_file)
    solution_savename = f'{param_dict["solver_type"]}_solution{save_tag}'
    speed_savename = f'{param_dict["solver_type"]}_shock_speed{save_tag}'
    Writer.create_folders(f'{param_dict["solver_type"]}', callbacks)
    solver_type = param_dict['solver_type']
    print(f"You are running the {solver_type} solver with the following parameters: {param_tag}.")
    print(f"The results will be saved in folders {solver_type}_pics, {solver_type}_data, and {solver_type}_video.")
    print("Give it a few seconds to compile necessary functions...")

    time, time_limit, time_out = [0.], param_dict['T'], 0.
    n_image = 0
    mesh = Mesh(param_dict['a'], param_dict['b'], param_dict['N'], 3)

    if param_dict['frame'] == 'LFOR':
        solver = Solver.create(param_dict['solver_type'], mesh, param_dict)
    elif param_dict['frame'] == 'SAFOR':
        solver = SAFORSolver.create(param_dict['solver_type'], mesh, param_dict)

    solver.set_initial_conditions(mesh, param_dict)

    timer_start = timer.perf_counter()
    while time[-1] < time_limit:
        if time[-1] >= time_out:
            if callbacks["write seconds"]:
                print(f't = {time[-1]:.2f}')
            if callbacks["write video"]:
                Writer.plot_solution(mesh, solver.phys_solution, time[-1], f'image{n_image:03d}')
            time_out += 0.01
            n_image += 1

        dt = solver.calculate_dt(solver.solution)
        solver.dt = dt if time[-1] + dt <= time_limit else time_limit-time[-1]
        if dt < 1e-15:
            break

        solver.timeintegrate()

        time.append(time[-1] + solver.dt)
        if np.any(np.isnan(solver.solution)):
            print('Floating point error')
            break

    #Last shot
    if callbacks["write video"]:
        Writer.plot_solution(mesh, solver.phys_solution, time[-1], f'image{n_image:03d}')

    wall_time = timer.perf_counter() - timer_start
    print(f'Simulation takes {wall_time:.2f} s.')

    if callbacks["plot final solution"]:
        Writer.plot_solution(mesh, solver.phys_solution, time[-1],
                             f, dpi=300)

    if callbacks["write final solution"]:
        Writer.write_solution(solver.phys_solution, solution_savename)

    if callbacks["write video"]:
        Writer.write_video(solver.phys_solution, solution_savename

    if callbacks["plot speed"]:
        try:
            Writer.plot_speed(time, solver.shock_speed, speed_savename, dpi=300)
        except:
            print(f"Unable to plot shock speed data for {param_dict['solver_type']} solver.")

    if callbacks["write speed"]:
        try:
            Writer.write_speed(time, solver.shock_speed, speed_savename)
        except:
            print(f"Unable to save shock speed data for {param_dict['solver_type']} solver.")


if __name__ == "__main__":
    simulate()
