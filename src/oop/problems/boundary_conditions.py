# Handle ghost point, including the moment after set_initial_conditions().
import numpy as np


def Det_BC_WENO5_jit(array, ambient, heat_release, D_next, gamma, beg, end):
    """
    Function for the boundary conditions, which MUST be stated in terms
    of physical variables
    We apply the rankine-Hugoniot conditions at the shock boundary (rigth) and
    the zero-gradient conditions at the left boundary.
    Parameters
    ----------
        array   : array_like
            Solution array in terms of the conserved variables
        ambient : array_like
            Ambient state variables at the shock
        D_next  : double
            Shock velocity to substitute into the Rankine-Hugoniot conditions
        beg     : int
            Index of the first grid cell after the left ghost cells
        end     : int
            Index of the last grid cell before the right ghost cells
    Returns
    -------
        array   : array_like
            Solution array with the applied boundary conditions
    """

def boundary_safor(array, ambient, heat_release, D_next, gamma, beg, end):
    rho_a, u_a, p_a, lambda_a = ambient[0], ambient[1], ambient[2], ambient[3]

    c_a = np.sqrt(gamma * p_a / rho_a)
    M_next = D_next / c_a
    p_s = (
        2.0 * gamma / (gamma + 1.0) * M_next * M_next
        - (gamma - 1.0) / (gamma + 1.0)
    ) * p_a
    rho_s = (
        (gamma + 1.0)
        * M_next
        * M_next
        / (2.0 + (gamma - 1.0) * M_next * M_next)
    ) * rho_a
    u_s = (2.0 / (gamma + 1.0) * (M_next * M_next - 1.0) / (M_next)) * c_a
    lambda_s = lambda_a

    shock_state_cons = np.empty(array.shape[0])
    shock_state_cons[0] = rho_s
    shock_state_cons[1] = u_s * rho_s
    shock_state_cons[3] = lambda_s * rho_s
    # Energy
    shock_state_cons[2] = (
        p_s / (gamma - 1)
        + 0.5 * shock_state_cons[1] * shock_state_cons[1] / rho_s
        - heat_release[-4] * shock_state_cons[3]
    )
    array[:, end - 1] = shock_state_cons

    for i in range(1, 4):
        # Left ghost points
        array[:, beg - i] = array[:, beg]
        # Right ghost points
        array[:, end - 1 + i] = array[:, end - 1]
    return array


def boundary_lfor_zero(mesh, array):
    array[:, mesh.domain[0]] = 0.
    array[:, mesh.domain[-1]] = 0.
    array[:, mesh.left_ghosts] = np.zeros(mesh.n_ghosts)
    array[:, mesh.right_ghosts] = np.zeros(mesh.n_ghosts)
    return array


def _boundary_lfor_zero_grad(mesh, array):
    array[:, mesh.domain[0]] = array[:, mesh.domain[1]]
    array[:, mesh.domain[-1]] = array[:, mesh.domain[-2]]
    array[:, mesh.left_ghosts] = array[:, mesh.domain[0]].reshape(-1,1)
    array[:, mesh.right_ghosts] = array[:,mesh.domain[-1]].reshape(-1,1)
    return array


def boundary_lfor_zero_grad(mesh):
    domain_min = mesh.domain[0]
    domain_max = mesh.domain[-1]
    left_ghosts_min = min(mesh.left_ghosts)
    left_ghosts_max = max(mesh.left_ghosts)
    right_ghosts_min = min(mesh.right_ghosts)
    right_ghosts_max = max(mesh.right_ghosts)
    def inner(array):
        array[:, domain_min] = array[:, domain_min+1]
        array[:, domain_max] = array[:, domain_max-1]
        n_rows = array.shape[0] if array.ndim > 1 else 1
        for i in range(n_rows):
            array[i, left_ghosts_min:left_ghosts_max] = array[i, domain_min]
            array[i, right_ghosts_min:right_ghosts_max] = array[i, domain_max]
        return array
    return inner


def boundary_periodic_right_going(mesh,array):
    array[:, mesh.domain[0]] = array[:, mesh.domain[-1]]
    array[:, mesh.left_ghosts] = array[:, mesh.domain[-mesh.n_ghosts-1:-1]]
    array[:, mesh.right_ghosts] = array[:, mesh.domain[1:mesh.n_ghosts+1]]
    return array


def boundary_periodic_left_going(mesh,array):
    array[:, mesh.domain[-1]] = array[:, mesh.domain[0]]
    array[:, mesh.left_ghosts] = array[:, mesh.domain[-mesh.n_ghosts-1:-1]]
    array[:, mesh.right_ghosts] = array[:, mesh.domain[1:mesh.n_ghosts+1]]
    return array


def boundary_none(mesh, array):
    return array
