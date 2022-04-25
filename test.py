#!/usr/bin/env python
# coding: utf-8
## EQUATION FOR D(T) was changed
# November 12, 2020
# # Problem statement
# This notebook should be used to modify, run, and debug the code for the problem of one-dimensional detonation propagating in the periodically varying ambient media.

# ![Problem.png](attachment:Problem.png)

# The governing equations are the reactive Euler equations in the shock-attached frame of reference [4,5]
#
# \begin{align}
# & \frac{\partial\rho}{\partial t} + \frac{\partial}{\partial x}\left( \rho (u-D)\right) = 0 \\
# & \frac{\partial}{\partial t} \left( \rho u\right) + \frac{\partial}{\partial x}\left( \rho u (u-D)+ p\right) = 0 \\
# & \frac{\partial}{\partial t} \left(\rho \left(e + \frac 12 u^2\right)\right) + \frac{\partial}{\partial x} \left(\rho (u-D) \left(e + \frac 12 u^2 \right) + u p\right)= 0 \\
# & \frac{\partial}{\partial t} \left( \rho \lambda \right)  + \frac{\partial}{\partial x}\left( \rho (u-D)\lambda \right) = K \rho (1-\lambda) \exp\left(-\frac{\rho E}{p}\right) \\
# \end{align}

# The Rankine-Hugoniot conditions are usual for these equations
# \begin{align}
#     & \rho_s \left(D-u_s\right) = \rho_a \left(D-u_a\right) \\
#     & p_s - p_a = \left(\rho_a \left(D - u_a\right)\right)^2 \left(\frac 1\rho_a -\frac 1\rho_s\right)\\
#     & e_s - e_a = \frac 12\left(p_s + p_a\right) \left(\frac 1\rho_a -\frac 1\rho_s\right)\\
#     & \lambda_s = \lambda_a
# \end{align}

# The shock-change equation for the shock velocity $D(t)$ is
# $$ \frac{dD}{dt} = - \left(\frac{d(\rho_s u_s)}{dD}\right)^{-1}\left.\left( \frac{\partial(\rho u (u-D)+p)}{\partial x} + D\frac{d\left(\rho_s u_s\right)}{d\xi}\right)\right|_{x=0} + \frac{c_f \rho u |u|}{2}$$

# The periodic conditions ahead of the shock are modeled in two ways presented below.
#
# RDE-like:
#
# \begin{align}
#     \rho_a(\xi) & = \frac{p_a}{R T_a + R A\left(1 + \sin\left(2 \pi k \xi\right)\right)} \\
#     u_a(\xi) & = 0\\
#     p_a(\xi) & = p_a\\
#     \lambda_a(\xi) & = A\left(1 + \sin\left(2 \pi k \xi\right)\right)
# \end{align}
# Periodic heat release:
# $$ Q_a(\xi) = Q(1 + A \sin(2 \pi k \xi)) $$

# The numerical algorithm is TVD Runge-Kutta (3rd order) + WENO5M spatial fluxes interpolation from [6].

# #### References
#
# [1] Harten, A. & Osher, S. (1987) Uniformly High-Order Accurate Nonoscillatory Schemes. I. SIAM J. Numer. Anal., 24, 279–309.
#
# [2] Jiang, G.-S. & Shu, C.-W. (1996) Efficient Implementation of Weighted ENO Schemes. Journal of Computational Physics, 126, 202–228.
#
# [3] Henrick, A.K., Aslam, T.D., & Powers, J.M. (2005) Mapped weighted essentially non-oscillatory schemes: Achieving optimal order near critical points. Journal of Computational Physics, 207, 542–567.
#
# [4] [Kasimov, A. R. & Stewart, D. S. On the dynamics of self-sustained one-dimensional detonations: A numerical study in the shock-attached frame. Physics of Fluids 16, 3566–3578 (2004).](https://www.researchgate.net/publication/32955598_On_the_dynamics_of_self-sustained_one-dimensional_detonations_A_numerical_study_in_the_shock-attached_frame)
#
# [5] [R. Semenko, L. Faria, A. Kasimov, B. Ermolaev, Set-valued solutions for non-ideal detonation, Shock Waves, 26(2), 141–160, 2016](https://link.springer.com/article/10.1007/s00193-015-0610-3)
#
# [6] [Henrick, A. K., Aslam, T. D. & Powers, J. M. Simulations of pulsating one-dimensional detonations with true fifth order accuracy. Journal of Computational Physics 213, 311–329 (2006).](https://www.sciencedirect.com/science/article/pii/S0021999105003827)

# # Code

# Description of the global variables (not to repeat in every function docstring):
# - `heat_release` : array_like
#         Array of the heat release (const or periodic space dependent)
# - `gamma` : double
#         Specific heat ratio
# - `D` : double
#         Shock speed
# - `step` : double
#         Spatial grid step size
# - `P` : double
#         Pressure
# - `rate_const` : double
#         Reaction rate constant
# - `act_energy` : double
#         Activation energy
# - `CFL` : double
#         Courant-Friedrichs-Lewy number

# In[1]:


import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import math
from scipy import (
    signal,
    interpolate,
    integrate,
)  # for averages and for PSD of detonation speed
import scipy.linalg as LA
from scipy.integrate import ode  # for ZND solution
import scipy.optimize as opt  # for nonlinear equations

# import tqdm.notebook as tqdm    # waiting bars
import time
import numba as nb  # for JIT compilation and accelaration
import h5py  # for storing data


# In[2]:


# get_ipython().run_line_magic('matplotlib', 'inline')


# # Numba JITted functions
# These are functions that are used in the `Simulation` class. The decorators `@nb.njit` and `@nb.vecorize` tell `numba` that the decorated function should be compiled. For more details, see [Numba documentation](https://numba.pydata.org/numba-doc/latest/index.html) with excellent guides and examples.

# In[3]:


@nb.njit("f8[:,:](f8[:,:],f8[:],f8,b1)")
def variable_transform_jit(array, heat_release, gamma, toConserved):
    """
    Transform an array of physical variables
    to an array of conserved variables and vice versa

    Parameters
    ----------
    array       : array_like
        An array of variables to be transformed
    toConserved : bool
        if True, converts to the conserved variables rho,m,E,Y
        if False, convertss to the physical variables rho,u, p, lam

    Returns
    -------
    transformed_array : array_like
        The result of the conversion
    """
    transformed_array = np.copy(array)
    if toConserved:  # from rho,u, p, lam to rho,m,E,Y
        transformed_array[1, :] = array[1, :] * array[0, :]
        transformed_array[3, :] = array[3, :] * array[0, :]
        # Energy
        transformed_array[2, :] = (
            array[2, :] / (gamma - 1)
            + 0.5
            * transformed_array[1, :]
            * transformed_array[1, :]
            / array[0, :]
            - heat_release * transformed_array[3, :]
        )

    else:  # from rho, m, E, Y to rho,u,p,lam
        transformed_array[1, :] = array[1, :] / array[0, :]
        transformed_array[3, :] = array[3, :] / array[0, :]
        # Pressure
        transformed_array[2, :] = (gamma - 1) * (
            array[2, :]
            - 0.5 * array[1, :] * array[1, :] / array[0, :]
            + heat_release * array[3, :]
        )
    return transformed_array


# In[4]:


@nb.njit("f8[:](f8[:],f8,f8,f8)")
def periodic_friction(nodes, displacement, eps, k_for_fric):
    period_part = np.zeros_like(nodes)
    period_part = np.where(
        nodes > -displacement,
        (1 + eps * np.sin(k_for_fric * (nodes + displacement))),
        np.zeros_like(nodes),
    )
    return period_part


# In[5]:


@nb.njit("f8(f8,f8,f8, f8[:], f8[:], f8[:], f8[:,:],f8[:], f8, f8, f8, f8,)")
def D_RHS_approx_jit(
    gamma,
    D,
    step,
    exact_fluxes,
    cur_state,
    cur_state_der,
    array,
    nodes,
    displacement,
    c_f,
    eps,
    k_for_fric,
):
    """
    Calculate the right hand side of the shock-change equation

    Parameters
    ----------
        exact_fluxes : array_like
            Momentum flux at the shock
        array: array_like
             previous solution for source term (need to take only state at shock)
        cur_state    : 4, array
            Current shock state

    Returns
    -------
        Value of the RHS
    """
    coef_friction = c_f * periodic_friction(
        nodes, displacement, eps, k_for_fric
    )
    source_term_momentum = (
        coef_friction[-3] * array[1, -3] * array[1, -3] / array[0, -3] / 2
    )
    # print(source_term_momentum)
    # The momentum flux gradient on the shock
    rho_a, u_a, p_a, lambda_a = (
        cur_state[0],
        cur_state[1],
        cur_state[2],
        cur_state[3],
    )
    m_flux = exact_fluxes  # HERE SHOULD BE ONLY MOMENTUM FLUX
    m_flux_dx = (
        -12.0 * m_flux[0]
        + 75.0 * m_flux[1]
        - 200.0 * m_flux[2]
        + 300.0 * m_flux[3]
        - 300.0 * m_flux[4]
        + 137.0 * m_flux[5]
    )
    m_flux_dx /= 60.0 * step
    vel_dif = D - u_a
    nom = (
        rho_a
        * vel_dif
        * (
            gamma * (rho_a * u_a * vel_dif - 2.0 * p_a)
            + rho_a * (2.0 * D**2 - 3.0 * D * u_a + u_a**2)
        )
    )
    denom = gamma * (2.0 * p_a + rho_a * vel_dif**2) - rho_a * vel_dif**2
    nom_by_denom_sq = nom / denom**2
    # The derivative of the momentum at the shock \rho*u|_s w.r.t. the shock velocity D
    nom_der = nom / vel_dif + rho_a * vel_dif * (
        gamma * rho_a * u_a + rho_a * (4.0 * D - 3.0 * u_a)
    )
    denom_der = 2.0 * rho_a * (gamma - 1.0) * vel_dif
    dD_dm = 1.0 / (nom_der / denom - denom_der * nom_by_denom_sq)
    # The derivative of the momentum at the shock \rho*u|_s w.r.t. the laboratory coordinate \xi
    dm_dxi = 0
    if cur_state_der[0] >= 1e-8:
        nom_der = nom / rho_a + rho_a * vel_dif * (
            gamma * (u_a * vel_dif) + (2.0 * D**2 - 3.0 * D * u_a + u_a**2)
        )
        denom_der = (gamma - 1.0) * vel_dif**2
        dm_drho = nom_der / denom - denom_der * nom_by_denom_sq
        dm_dxi += dm_drho * cur_state_der[0]
    if cur_state_der[1] >= 1e-8:
        nom_der = -nom / vel_dif + rho_a**2 * vel_dif * (
            (gamma - 3.0) * D - (gamma - 1.0) * 2.0 * u_a
        )
        denom_der = 2.0 * (1.0 - gamma) * rho_a * vel_dif
        dm_du = nom_der / denom - denom_der * nom_by_denom_sq
        dm_dxi += dm_du * cur_state_der[1]
    if cur_state_der[2] >= 1e-8:
        nom_der = -2.0 * rho_a * vel_dif * gamma
        denom_der = 2.0 * gamma
        dm_dp = nom_der / denom - denom_der * nom_by_denom_sq
        dm_dxi += dm_dp * cur_state_der[2]
    return -dD_dm * (m_flux_dx + D * dm_dxi + source_term_momentum)


# In[6]:


@nb.njit("f8[:,:](f8[:,:],f8[:], f8)")
def Det_ef_jit(array, P, D):
    # ef = exact flux
    """
    Calculate a matrix of exact flux for the reactive Euler equations in SAFoR
    from the array (which is also matrix) of variables given at the mesh nodes
    Need for the conserved variables.
    Parameters
    ----------
        array : array_like
            Conserved variables
            array[0,:] - rho
            array[1,:] - rho*u
            array[2,:] - E
            array[3,:] - rho*lambda
    Returns
    -------
        flux_array : array_like
            Calculated array of the exact fluxes
    """
    flux_array = np.empty_like(array)
    flux_array[0, :] = array[1, :]
    flux_array[1, :] = array[1, :] * array[1, :] / array[0, :] + P
    flux_array[2, :] = array[1, :] * (array[2, :] + P) / array[0, :]
    flux_array[3, :] = array[1, :] * array[3, :] / array[0, :]
    # Crucial for the shock-attached frame
    flux_array -= D * array
    return flux_array


# In[7]:


@nb.vectorize("f8(f8)")
def mexp_jit(x):
    """
    Numba optimized exponent
    """
    return math.exp(x)


# In[8]:


@nb.njit(
    "Tuple((f8[:,:], f8[:]))(f8[:,:], f8[:], f8, f8, f8[:], f8, f8, f8, f8)"
)
def Det_source_jit(
    array, P, act_energy, rate_const, nodes, displacement, c_f, eps, k_for_fric
):
    """
    Calculate the RHS for the reactive Euler equations
    Parameters
    ----------
        array : array_like
            Conserved variables
    Returns
    -------
        source : array_like
            The right hand side for the reactive Euler equations
    """
    # array.shape[1]
    coef_friction = c_f * periodic_friction(
        nodes, displacement, eps, k_for_fric
    )
    # c_f = array[1, :]
    source = np.zeros_like(array)
    source[1, :] = -coef_friction * array[1, :] * array[1, :] / array[0, :] / 2
    source[-1, :] = (
        rate_const
        * (array[0, :] - array[3, :])
        * mexp_jit(-act_energy * array[0, :] / P)
    )
    return source, coef_friction


# In[9]:


@nb.njit(
    "Tuple((f8[:,:], f8[:,:],f8[:,:], f8,f8[:]))(f8[:,:], f8[:], f8, f8, f8, f8, f8,f8, f8[:],f8,f8,f8,f8)"
)
def Det_LF_jit(
    array,
    heat_release,
    step,
    D,
    gamma,
    act_energy,
    rate_const,
    CFL,
    nodes,
    displacement,
    c_f,
    eps,
    k_for_fric,
):
    # LF = Lax-Friedrichs splitting
    """
    Returns the EXACT eigenvalues and the EXACT right eigenvectors system for
    the Roe averaged Jacobian matrix. The eigenvalues are returned as a matrix of the size
    (NoV x total_num). The eigenvectors: (NoV x NoV x total_num)
    There are two versions of the entropy fix (not used here):
    1. with eigenvalues
    2. with fluxes

    Parameters
    ----------
        Described above

    Returns
    -------
        exact_fluxes :  array calculated with Det_ef_jit function
        Evals        :  array of the values of the eigenvalues along the grid
        RHS          :  the right hand side of the reactive Euler
                        equations calculated with Det_source_jit
        time_step    :  numerical time determined with the CFL condition
    """
    u = array[1, :] / array[0, :]
    lam = array[3, :] / array[0, :]
    P = (gamma - 1.0) * (
        array[2, :]
        - 0.5 * array[1, :] * array[1, :] / array[0, :]
        + array[3, :] * heat_release
    )
    H = (array[2, :] + P) / array[0, :]
    c = np.sqrt((gamma - 1.0) * (H - 0.5 * u * u))

    Evals = np.empty_like(array)
    Evals[0, :] = u - c
    Evals[1, :] = u
    Evals[2, :] = u
    Evals[3, :] = u + c
    Evals = Evals - D

    time_step = CFL * (step / np.max((np.abs(Evals))))
    exact_fluxes = Det_ef_jit(array, P, D)
    # RHS = Det_source_jit(array, P, act_energy, rate_const)
    RHS, c_fric = Det_source_jit(
        array,
        P,
        act_energy,
        rate_const,
        nodes,
        displacement,
        c_f,
        eps,
        k_for_fric,
    )
    return exact_fluxes, Evals, RHS, time_step, c_fric


# In[10]:


@nb.njit("f8[:,:](f8[:,:], f8[:], f8[:], f8, f8, i4, i4)")
def Det_BC_WENO5_jit(array, ambient, heat_release, D_next, gamma, beg, end):
    """
    Function for the boundary conditions, which MUST be stated in terms
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


# In[11]:


# Value to ensure the non-zero denominator in the smoothness indicators
eps_forWENO = 1e-40


@nb.vectorize("f8(f8,f8)", nopython=True)
def g_numba(w_k, omega):
    """
    Calculate the weighted coefficients according to [3,6]
    """
    return (
        omega
        * (w_k + w_k**2 - 3 * w_k * omega + omega**2)
        / (w_k**2 + (1 - 2.0 * w_k) * omega)
    )


@nb.njit("f8[:,:](f8[:,:], f8[:,:], f8[:,:], f8[:,:], f8[:,:])")
def WENO_flux_interpolation_jit(nodei_2, nodei_1, nodei, nodeip1, nodeip2):
    """
    Implement the calculation of the numerical fluxes using modified weighted
    essentially non-oscillatory method of the fifth order from [3] with three
    stencils defined by the nodes i-2, i-1, i, i+1, i+2 for right (positive) directed waves
    and by the nodes i+3, i+2, i+1, i, i-1 for left (negative) directed waves
    Parameters
    ----------
        nodei : the middle node in the 5 point stencil for WENO
        nodei_2, nodei_1 : left nodes (e.g. i-2, i-1)
        nodeip1, nodeip2 : right nodes (e.g. i+1, i+2)
    Returns
    -------
        modified_weno5_flux : array_like
            The calculated approximation of the flux between two cells
    """
    # Interpolation of the fluxes
    q_im1 = 1.0 / 3.0 * nodei_2 - 7.0 / 6.0 * nodei_1 + 11.0 / 6.0 * nodei
    q_i = -1.0 / 6.0 * nodei_1 + 5.0 / 6.0 * nodei + 1.0 / 3.0 * nodeip1
    q_ip1 = 1.0 / 3.0 * nodei + 5.0 / 6.0 * nodeip1 - 1.0 / 6.0 * nodeip2
    # Indicators of smoothness
    IS_im1 = (
        13.0 / 12.0 * (nodei_2 - 2.0 * nodei_1 + nodei) ** 2
        + 0.25 * (nodei_2 - 4.0 * nodei_1 + 3.0 * nodei) ** 2
    )
    IS_i = (
        13.0 / 12.0 * (nodei_1 - 2.0 * nodei + nodeip1) ** 2
        + 0.25 * (nodei_1 - nodeip1) ** 2
    )
    IS_ip1 = (
        13.0 / 12.0 * (nodei - 2.0 * nodeip1 + nodeip2) ** 2
        + 0.25 * (3.0 * nodei - 4.0 * nodeip1 + nodeip2) ** 2
    )
    # Weights for stencils
    alpha_im1 = 1.0 / 10.0 / (eps_forWENO + IS_im1) ** 2
    alpha_i = 6.0 / 10.0 / (eps_forWENO + IS_i) ** 2
    alpha_ip1 = 3.0 / 10.0 / (eps_forWENO + IS_ip1) ** 2
    alpha_sum = alpha_im1 + alpha_i + alpha_ip1
    # Modified weights
    mod_omega_0 = g_numba(1.0 / 10.0, alpha_im1 / alpha_sum)
    mod_omega_1 = g_numba(6.0 / 10.0, alpha_i / alpha_sum)
    mod_omega_2 = g_numba(3.0 / 10.0, alpha_ip1 / alpha_sum)
    mod_omega_sum = mod_omega_0 + mod_omega_1 + mod_omega_2
    # Numerical flux # Different schemes
    # weno5_flux = (alpha_im1*q_im1 + alpha_i*q_i + alpha_ip1*q_ip1)/alpha_sum
    # upstream_central = 1/60*(2*nodei_2 - 13*nodei_1 +47* nodei +27*nodeip1-3.*nodeip2)
    modified_weno5_flux = (
        mod_omega_0 * q_im1 + mod_omega_1 * q_i + mod_omega_2 * q_ip1
    ) / mod_omega_sum
    return modified_weno5_flux


# In[12]:


@nb.njit(
    "Tuple((f8[:,:], f8, f8,f8[:]))(f8[:,:], f8[:], f8[:],f8[:], f8, f8, f8, f8, f8, f8, i4,i4,i4,f8[:],f8,f8,f8,f8)"
)
def spatial_DiffOp_jit(
    array,
    heat_release,
    ambient,
    ambient_der,
    step,
    D,
    gamma,
    act_energy,
    rate_const,
    CFL,
    start,
    finish,
    end,
    nodes,
    displacement,
    c_f,
    eps,
    k_for_fric,
):
    """
    The discretization of the spatial differential operator calculates
    the numerical flux for values written in the array according to
    Lax-Friedrichs Splitting 1D WENO5M scheme from [3]
    Parameters
    ----------
        ambient     : vector
            State variables right ahead of the shock
        ambient_der : vector
            Their spatial derivatives
        start       : int
            First index for the flux approximations
        finish      : int
            Last index for the flux approximations
        end         : int
            Index of the last grid cell before the right ghost cells

    Returns
    -------
        RHS_for_RK : array_like
            Approximation of the spatial differential operator that is the RHS for
            the time ODE for TVD Runge-Kutta method
        dD_dt      : double
            The right hand side of the shock-change equation
        tau        : double
            Current time step

    """
    ex_fluxes, Evals, RHS, tau, c_fric = Det_LF_jit(
        array,
        heat_release,
        step,
        D,
        gamma,
        act_energy,
        rate_const,
        CFL,
        nodes,
        displacement,
        c_f,
        eps,
        k_for_fric,
    )
    fluxes = ex_fluxes[:, :-1]
    # Nodes 0<=i<=N-3=end-4=finish-1 Section 3.2.2 of [6]
    # LF splitting
    alpha_mat_for_max = np.maximum(np.abs(Evals[:, :-1]), np.abs(Evals[:, 1:]))
    alpha_vec = np.empty(alpha_mat_for_max.shape[1])
    for i in nb.prange(alpha_vec.shape[0]):
        alpha_vec[i] = np.max(alpha_mat_for_max[:, i])
    alpha_u = array[:, :-1] * alpha_vec
    fluxes_plus = fluxes + alpha_u
    fluxes_minus = fluxes - alpha_u
    # Fluxes for waves in positive x direction
    nodei_2 = fluxes_plus[:, start - 2 : finish - 2]
    nodei_1 = fluxes_plus[:, start - 1 : finish - 1]
    nodei = fluxes_plus[:, start:finish]
    nodeip1 = fluxes_plus[:, start + 1 : finish + 1]
    nodeip2 = fluxes_plus[:, start + 2 : finish + 2]
    WENO5M_pos = WENO_flux_interpolation_jit(
        nodei_2, nodei_1, nodei, nodeip1, nodeip2
    )
    # Fluxes for waves in negative x direction
    nodei_2 = fluxes_minus[:, start + 3 : finish + 3]
    nodei_1 = fluxes_minus[:, start + 2 : finish + 2]
    nodei = fluxes_minus[:, start + 1 : finish + 1]
    nodeip1 = fluxes_minus[:, start:finish]
    nodeip2 = fluxes_minus[:, start - 1 : finish - 1]
    WENO5M_neg = WENO_flux_interpolation_jit(
        nodei_2, nodei_1, nodei, nodeip1, nodeip2
    )
    # Final numerical flux
    num_flux = 0.5 * (WENO5M_pos + WENO5M_neg)
    # Numerical conservative flux derivative
    flux_der_for_domain = (num_flux[:, 1:] - num_flux[:, :-1]) / step
    # Preshock state Section 3.2.3 of [6]
    flux_der_preshock_2 = (
        (
            -2.0 * fluxes[:, end - 6]
            + 15.0 * fluxes[:, end - 5]
            - 60.0 * fluxes[:, end - 4]
            + 20.0 * fluxes[:, end - 3]
            + 30.0 * fluxes[:, end - 2]
            - 3.0 * fluxes[:, end - 1]
        )
        / 60.0
        / step
    )
    # Preshock state Section 3.2.3 of [6]
    flux_der_preshock_1 = (
        (
            -fluxes[:, end - 5]
            + 6.0 * fluxes[:, end - 4]
            - 18.0 * fluxes[:, end - 3]
            + 10.0 * fluxes[:, end - 2]
            + 3.0 * fluxes[:, end - 1]
        )
        / 12.0
        / step
    )
    # Numerical conservative flux derivative for the nodes up to the shock
    flux_der = np.hstack(
        (
            flux_der_for_domain,
            flux_der_preshock_2.reshape(-1, 1),
            flux_der_preshock_1.reshape(-1, 1),
        )
    )
    dD_dt = D_RHS_approx_jit(
        gamma,
        D,
        step,
        ex_fluxes[1, end - 6 : end],
        ambient,
        ambient_der,
        array,
        nodes,
        displacement,
        c_f,
        eps,
        k_for_fric,
    )
    RHS_for_RK = RHS[:, start + 1 : finish + 2] - flux_der
    return RHS_for_RK, dD_dt, tau, c_fric


# In[13]:


@nb.njit("f8(f8[:],f8[:])")
def dot_numba(coefs, values):
    _sum = 0.0
    for c, v in zip(coefs, values):
        _sum += c * v
    return _sum


@nb.njit(
    "Tuple((f8[:,:], f8, f8,f8[:],f8))(i4, f8[:,:],f8[:], f8[:], f8[:], f8, f8, f8, f8, f8, f8, i4,i4,i4,i4, f8[:],f8,f8,f8,f8)"
)
def RK(
    Runge_Kutta,
    solution,
    heat_release,
    ambient,
    ambient_der,
    step,
    D_cur,
    gamma,
    act_energy,
    rate_const,
    CFL,
    start,
    finish,
    beg,
    end,
    nodes,
    displacement,
    c_f,
    eps,
    k_for_fric,
):
    """
    Total variation diminishing Runge-Kutta method for time integration

    Parameters
    ----------
        Runge_Kutta : int
            Order of the method (currently, only 3rd is possible)
        solution    : array_like
            Solution on the nth time step
        D_cur       : double
            Current value of the shock speed on the nth time step
        Other parameters are described inside the corresponding functions

    Returns
    -------
        update_solution : array_like
            Solution on the (n+1)th time step
        D_next          : double
            Value of the shock speed on the (n+1)th time step
        tau             :
            time step = t_{n+1} - t_{n} (CFL condition)
    """
    SDO_approx, dD_dt, tau, c_fric = spatial_DiffOp_jit(
        solution,
        heat_release,
        ambient,
        ambient_der,
        step,
        D_cur,
        gamma,
        act_energy,
        rate_const,
        CFL,
        start,
        finish - 2,
        end,
        nodes,
        displacement,
        c_f,
        eps,
        k_for_fric,
    )
    update_solution = np.empty_like(solution)
    update_solution[:, start + 1 : finish] = (
        solution[:, start + 1 : finish] + tau * SDO_approx
    )
    D_next = D_cur + tau * dD_dt
    # Boundary conditions
    update_solution = Det_BC_WENO5_jit(
        update_solution, ambient, heat_release, D_next, gamma, beg, end
    )
    if Runge_Kutta == 3:
        """3rd order Runge-Kutta time discretization"""
        ### (Shu & Osher, 1988)
        update_solution2 = np.empty_like(solution)
        SDO_approx2, dD_dt2, _, c_fric2 = spatial_DiffOp_jit(
            update_solution,
            heat_release,
            ambient,
            ambient_der,
            step,
            D_next,
            gamma,
            act_energy,
            rate_const,
            CFL,
            start,
            finish - 2,
            end,
            nodes,
            displacement,
            c_f,
            eps,
            k_for_fric,
        )
        D_1 = D_cur + tau * (0.25 * dD_dt + 0.25 * dD_dt2)
        update_solution2[:, start + 1 : finish] = solution[
            :, start + 1 : finish
        ] + tau * (0.25 * SDO_approx + 0.25 * SDO_approx2)
        update_solution2 = Det_BC_WENO5_jit(
            update_solution2, ambient, heat_release, D_1, gamma, beg, end
        )

        SDO_approx3, dD_dt3, _, c_fric3 = spatial_DiffOp_jit(
            update_solution2,
            heat_release,
            ambient,
            ambient_der,
            step,
            D_1,
            gamma,
            act_energy,
            rate_const,
            CFL,
            start,
            finish - 2,
            end,
            nodes,
            displacement,
            c_f,
            eps,
            k_for_fric,
        )
        D_next = D_cur + tau * (1 / 6 * dD_dt + 1 / 6 * dD_dt2 + 2 / 3 * dD_dt3)
        update_solution[:, start + 1 : finish] = solution[
            :, start + 1 : finish
        ] + tau * (
            1 / 6 * SDO_approx + 1 / 6 * SDO_approx2 + 2 / 3 * SDO_approx3
        )
        update_solution = Det_BC_WENO5_jit(
            update_solution, ambient, heat_release, D_next, gamma, beg, end
        )
        D_acc = 1 / 6 * dD_dt + 1 / 6 * dD_dt2 + 2 / 3 * dD_dt3
    if Runge_Kutta == 5:
        """5th order Runge-Kutta time discretization"""
        ### (Henrick, Aslam & Powers, 2006)
        j4_coefs = np.array(
            [2046.0 / 15625.0, -454.0 / 15625.0, 1533.0 / 15625.0]
        )
        j5_coefs = np.array(
            [-739.0 / 5625.0, 511.0 / 5625.0, -566.0 / 16875.0, 20.0 / 27.0]
        )
        j6_coefs = np.array(
            [
                11822.0 / 21875.0,
                -6928.0 / 21875.0,
                -4269.0 / 21875.0,
                -4.0 / 7.0,
                54.0 / 35.0,
            ]
        )
        b_coefs = np.array(
            [1.0 / 24.0, 0.0, 0.0, 125.0 / 336.0, 27.0 / 56.0, 5.0 / 48.0]
        )

        update_solution2 = np.empty_like(solution)

        SDO_approx2, dD_dt2, _, c_fric2 = spatial_DiffOp_jit(
            update_solution,
            heat_release,
            ambient,
            ambient_der,
            step,
            D_next,
            gamma,
            act_energy,
            rate_const,
            CFL,
            start,
            finish - 2,
            end,
            nodes,
            displacement,
            c_f,
            eps,
            k_for_fric,
        )
        D_1 = D_cur + tau * (0.25 * dD_dt + 0.25 * dD_dt2)
        update_solution2[:, start + 1 : finish] = solution[
            :, start + 1 : finish
        ] + tau * (0.25 * SDO_approx + 0.25 * SDO_approx2)
        update_solution2 = Det_BC_WENO5_jit(
            update_solution2, ambient, heat_release, D_1, gamma, beg, end
        )

        SDO_approx3, dD_dt3, _, c_fric3 = spatial_DiffOp_jit(
            update_solution2,
            heat_release,
            ambient,
            ambient_der,
            step,
            D_1,
            gamma,
            act_energy,
            rate_const,
            CFL,
            start,
            finish - 2,
            end,
            nodes,
            displacement,
            c_f,
            eps,
            k_for_fric,
        )
        D_2 = D_cur + tau * dot_numba(
            j4_coefs, np.array([dD_dt, dD_dt2, dD_dt3])
        )
        update_solution2[:, start + 1 : finish] = solution[
            :, start + 1 : finish
        ] + tau * (
            j4_coefs[0] * SDO_approx
            + j4_coefs[1] * SDO_approx2
            + j4_coefs[2] * SDO_approx3
        )
        update_solution2 = Det_BC_WENO5_jit(
            update_solution2, ambient, heat_release, D_2, gamma, beg, end
        )

        SDO_approx4, dD_dt4, _, c_fric4 = spatial_DiffOp_jit(
            update_solution2,
            heat_release,
            ambient,
            ambient_der,
            step,
            D_2,
            gamma,
            act_energy,
            rate_const,
            CFL,
            start,
            finish - 2,
            end,
            nodes,
            displacement,
            c_f,
            eps,
            k_for_fric,
        )
        D_3 = D_cur + tau * dot_numba(
            j5_coefs, np.array([dD_dt, dD_dt2, dD_dt3, dD_dt4])
        )
        update_solution2[:, start + 1 : finish] = solution[
            :, start + 1 : finish
        ] + tau * (
            j5_coefs[0] * SDO_approx
            + j5_coefs[1] * SDO_approx2
            + j5_coefs[2] * SDO_approx3
            + j5_coefs[3] * SDO_approx4
        )
        update_solution2 = Det_BC_WENO5_jit(
            update_solution2, ambient, heat_release, D_3, gamma, beg, end
        )

        SDO_approx5, dD_dt5, _, c_fric5 = spatial_DiffOp_jit(
            update_solution2,
            heat_release,
            ambient,
            ambient_der,
            step,
            D_3,
            gamma,
            act_energy,
            rate_const,
            CFL,
            start,
            finish - 2,
            end,
            nodes,
            displacement,
            c_f,
            eps,
            k_for_fric,
        )
        D_4 = D_cur + tau * dot_numba(
            j6_coefs, np.array([dD_dt, dD_dt2, dD_dt3, dD_dt4, dD_dt5])
        )
        update_solution2[:, start + 1 : finish] = solution[
            :, start + 1 : finish
        ] + tau * (
            j6_coefs[0] * SDO_approx
            + j6_coefs[1] * SDO_approx2
            + j6_coefs[2] * SDO_approx3
            + j6_coefs[3] * SDO_approx4
            + j6_coefs[4] * SDO_approx5
        )
        update_solution2 = Det_BC_WENO5_jit(
            update_solution2, ambient, heat_release, D_4, gamma, beg, end
        )

        SDO_approx6, dD_dt6, _, c_fric6 = spatial_DiffOp_jit(
            update_solution2,
            heat_release,
            ambient,
            ambient_der,
            step,
            D_4,
            gamma,
            act_energy,
            rate_const,
            CFL,
            start,
            finish - 2,
            end,
            nodes,
            displacement,
            c_f,
            eps,
            k_for_fric,
        )

        D_acc = dot_numba(
            b_coefs, np.array([dD_dt, dD_dt2, dD_dt3, dD_dt4, dD_dt5, dD_dt6])
        )
        D_next = D_cur + tau * D_acc
        update_solution[:, start + 1 : finish] = solution[
            :, start + 1 : finish
        ] + tau * (
            b_coefs[0] * SDO_approx
            + b_coefs[1] * SDO_approx2
            + b_coefs[2] * SDO_approx3
            + b_coefs[3] * SDO_approx4
            + b_coefs[4] * SDO_approx5
            + b_coefs[5] * SDO_approx6
        )
        update_solution = Det_BC_WENO5_jit(
            update_solution, ambient, heat_release, D_next, gamma, beg, end
        )
    return update_solution, D_next, tau, c_fric, D_acc


# In[14]:


class Cell1D:
    """
    1-D computational cell
    """

    def __init__(self, xc=0.0, lenght=1.0, index=0):
        """
        Initializes a cell.

        Sets the center point and the lenght of the cell. Then calculates its
        left and right end points. Index of the cell corresponds
        to the mesh index of its center point.
        Initializes the initial and current values in the cell center to zero.

        Parameters
        ----------
            xc     : float
                x-coordinate of the cell center.
            lenght : float
                lenght of the cell.
            index  : int
                mesh index of xc.
        """
        self.xc = xc
        self.lenght = lenght
        self.index = index

        # End points
        self.xl, self.xr = xc - 0.5 * lenght, xc + 0.5 * lenght
        # Initial and current values at xc
        self.init_val = 0.0
        self.val = 0.0
        # Flux of a cell
        self.flux = {index + 1 / 2: 0.0}

    def average(self, func):
        """
        Method to find the average of some function over the cell

        Parameters
        ----------
        func : function to integrate
        """

        def vec_func(x):
            return np.array(func(x)).reshape((-1, 1))

        num_of_var = vec_func(self.xc).shape[0]
        cell_average = np.empty((num_of_var, 1))
        for i in range(num_of_var):

            def func1d(x):
                return vec_func(x)[i, :]

            cell_average[i], _ = integrate.quad(func1d, self.xl, self.xr)
        return cell_average / self.lenght


class Mesh1D:
    """
    Define a computational mesh with cells.
    Setters and getters work with array of shape (num_var, total_num)
    Could be replaced by usual np.linspace or np.arange.
    Attempt to use OOP approach to the code failed after the necessity of optimization
    and @nb.njit decorator that works only for functions.
    That is why there are some ugliness and mess in the above function definitions.
    """

    def __init__(self, x_0=0.0, x_N=1.0, num_nodes=2, num_ghost=0):
        """
        Create an array of cells with num_nodes nodes from x_0 to x_N
        and num_ghost of ghost nodes left to x_0 and right to x_N

        Parameters
        ----------
            x_0: float
                left end node of the mesh
            X_N: float
                right end node of the mesh
            num_nodes: int
                number of nodes in the mesh {x_i} i=0,N
            num_ghost: int
                number of ghost nodes
        """

        self.x_0 = x_0
        self.x_N = x_N
        self.NoN = num_nodes
        self.NoG = num_ghost
        # Total number of nodes or cells
        self.total_num = num_nodes + 2 * num_ghost
        self.N = num_nodes - 1  # real index of the right endpoint
        self.beg = num_ghost  # numpy index of the left endpoint
        self.end = (
            num_ghost + num_nodes
        )  # numpy index next to the right endpoint + 1
        # domain = [beg:end]

        if x_N > x_0:
            self.step = (x_N - x_0) / self.N
        else:
            raise Exception(
                "The right endpoint is less than or equal to the left endpoint"
            )

        self.cells = np.empty(self.total_num, dtype=Cell1D)
        for i, xc in enumerate(
            np.linspace(
                x_0 - self.NoG * self.step,
                x_N + self.NoG * self.step,
                self.total_num,
            )
        ):
            self.cells[i] = Cell1D(xc, self.step, i - self.NoG)

        self.domain = self.cells[self.beg : self.end]

    def get_nodes(self, domain=False):
        """
        Getter of an array of the cell centers coordinates

        Parameters
        ----------
        domain: bool
            Whether to return the values without ghosts or not
        """
        if domain:
            return np.array([cell.xc for cell in self.domain])
        else:
            return np.array([cell.xc for cell in self.cells])

    def get_values(self, domain=False):
        """
        Getter of values written in the nodes. Returns the matrix that has
        (m x n) size with m variables and n = N+1 or total_num nodes like this
        |variables\nodes|...|x_0||x_1|...|...|x_N|...|
                    |u_1|...|v10||v11|...|...|...|...|
                    |u_2|...|v20||...|...|...|...|...|
                    |...|...|...||...|...|...|...|...|
                    |u_m|...|...||...|...|...|vmn|...|,
        where vij is the value of the i-th variable at the node x_j

        Parameters
        ----------
        domain: bool
            Whether to return the values without ghosts or not
        """
        if domain:
            return np.hstack([cell.val for cell in self.domain])
        else:
            return np.hstack([cell.val for cell in self.cells])

    def get_fluxes(self, domain=True):
        """
        Getter of fluxes attributed to each cell. Returns the matrix that has
        (m x n) size with m variables and n = N+1 or total_num nodes like this
        |variables\nodes|...|x_0||x_1|...|...|x_N|...|
                    |u_1|...|f10||f11|...|...|...|...|
                    |u_2|...|f20||...|...|...|...|...|
                    |...|...|...||...|...|...|...|...|
                    |u_m|...|...||...|...|...|fmn|...|,
        where fij is the flux of the i-th variable at the point x_{j+1/2}
        Examples:
        mym = Mesh1D(num_nodes=5, num_ghost=2)
        print(mym)
        mym.plot_mesh(domain=False)
        arr_for_flux = np.arange(mym.total_num)
        arr_for_flux3 = np.vstack((arr_for_flux, arr_for_flux+1, arr_for_flux+3))
        mym.set_fluxes(arr_for_flux3)
        for i in range(mym.total_num):
            print(mym.cells[i].flux)
        a = mym.get_fluxes(domain=False)
        print(a)
        a.shape

        Parameters
        ----------
        domain: bool
            Whether to return the values without ghosts or not
        """
        self.fluxes = {}
        for cell in self.cells:
            self.fluxes.update(cell.flux)
        if domain:
            flux_array = np.hstack(
                list(self.fluxes.values())[self.beg : self.end]
            )
        else:
            flux_array = np.hstack(list(self.fluxes.values()))
        return flux_array

    def set_values(self, init, with_averages=False, args=()):
        """
        Setter for the values in the mesh nodes. Set the vector of the size
        (m x 1) with m variables in total_num nodes.

        Parameters
        ----------
        init: numpy.ndarray or function
            An array or a function to set the values in mesh nodes.
            If array, should be of the size (m x total_num).
        """
        if callable(init):
            if with_averages:
                for i in range(self.total_num):
                    self.cells[i].val = self.cells[i].average(init)
            else:
                for i in range(self.total_num):
                    self.cells[i].val = init(self.cells[i].xc, *args).reshape(
                        (-1, 1)
                    )
        elif type(init) == np.ndarray:
            if len(init.shape) == 1:
                init = init.reshape((1, -1))
            if init.shape[1] == self.total_num:
                for i in range(self.total_num):
                    self.cells[i].val = init[:, i].reshape((-1, 1))
            else:
                raise Exception(
                    "Input array shape: {}, when the mesh size is {}".format(
                        init.shape[1], self.total_num
                    )
                )
        else:
            raise Exception(
                "An array or a function is allowed for setting the values"
            )
        return self

    def set_fluxes(self, init):
        """
        Setter for the cell fluxes

        Parameters
        ----------
        init: numpy.ndarray or function
            An array or a function to set the fluxes.
            If array, should be of the size (m x total_num).
        """
        if type(init) == np.ndarray:
            if len(init.shape) == 1:
                init = init.reshape((1, -1))
            if init.shape[1] == self.total_num:
                for i in range(self.total_num):
                    self.cells[i].flux = {
                        self.cells[i].index + 1 / 2: init[:, i].reshape((-1, 1))
                    }
            else:
                raise Exception(
                    "Input array shape is not consistent with the mesh size"
                )
        else:
            raise Exception("An array is allowed for setting the fluxes")

    def plot_mesh(self, domain=True, with_fluxes=False):
        """
        Method for drawing the cell centers and the
        boundaries where the fluxes are defined

        Parameters
        ----------
            domain      : bool
                Whether or not to depict the ghost nodes
            with_fluxes : bool
                Make arrows for the fluxes at the boundaries or not

        Returns
        -------
            Plot of the mesh
        """
        if domain:
            cells = self.domain
        else:
            cells = self.cells
        fig, ax = plt.subplots(figsize=(10, 5))
        for cell in cells:
            ax.scatter(cell.xl, 0.0, color="b")
        ax.scatter(cells[-1].xr, 0.0, color="b", label="Cell borders")
        for cell in cells:
            ax.scatter(cell.xc, 0.0, color="r")
        ax.scatter(cells[-1].xc, 0.0, color="r", label="Cell centers")
        #         if with_fluxes:
        #             for cell in cells[1:]:
        #                 ax.quiver(cell.xr, 0, list(cell.flux.values())[0], 0.)
        #             q = ax.quiver(cells[0].xr, 0, list(cells[0].flux.values())[
        #                           0], 0, label='Cell fluxes')
        ax.grid()
        ax.legend(loc="best")
        plt.show()

    def plot_vals(self, exact=False, domain=True, labels=None, save=False):
        """
        Method for drawing profile of the values defined on the mesh.
        Simply, plot (x, f(x)) where x are centers of the cells
        and f(x) are the prescribed values.

        Parameters
        ----------
            exact       : bool
                If false, make the scatter plot
            domain      : bool
                Whether or not to depict the ghost nodes
            labels      : dict_like
                Titles for the plots
            save        : bool
                If True, save the plots in files

        Returns
        -------
            Plot of the values over the mesh
        """
        values = self.get_values(domain=domain)
        if len(values.shape) == 1:
            values = values.reshape((1, -1))
        if not labels:
            labels = range(values.shape[0])
        for i in range(values.shape[0]):
            fig, ax = plt.subplots(figsize=(10, 10))
            if exact:
                ax.plot(
                    self.get_nodes(domain=domain),
                    values[i, :],
                    label=labels[i],
                    color="blue",
                )
            else:
                ax.scatter(
                    self.get_nodes(domain=domain),
                    values[i, :],
                    label=labels[i],
                    color="darkorange",
                )
            ax.set_xlabel("x")
            ax.grid()
            ax.legend(loc="best")
            if save:
                plt.savefig(str(labels[i]) + ".eps", format="eps")
            plt.show()


# In[15]:


def calc_initial(file, L, N, all_nodes, NoG, gamma, Q, act_energy):
    """
    Function for calculating the ZND solution on the mesh of the lenght L
    with N nodes and writing it in the hdf5 file

    Parameters
    ----------
        file      : hdf5 file
            File descriptor to write the initial solution
        L         : double
            Length of the computational domain
        N         : int
            Number of nodes in the mesh
        all_nodes : array_like
            Centers of the cells in the mesh
        NoG       : int
            Number of the ghost points
        Q         : double
            Average heat release to calculate the ZND solution

    Returns
    -------
        rate_const  : double
            Rate of the chemical reaction scaled in such a way that
            the half reaction length (\lambda = 0.5) is equalt to 1
        init_array  : array_like
            Initial ZND profile
    """

    def for_reac_rate(y):
        D_CJ = np.sqrt(gamma + (gamma * gamma - 1.0) * Q / 2.0) + np.sqrt(
            (gamma * gamma - 1.0) * Q / 2.0
        )
        V_lam = (
            gamma
            / D_CJ
            / D_CJ
            * (1.0 + D_CJ * D_CJ)
            / (gamma + 1.0)
            * (
                1.0
                - (D_CJ * D_CJ - gamma)
                / (1.0 + D_CJ * D_CJ)
                / gamma
                * (np.sqrt(1 - y))
            )
        )
        u_lam = (
            1
            / (gamma + 1.0)
            * (D_CJ * D_CJ - gamma)
            / D_CJ
            * (1 + np.sqrt(1 - y))
        )
        p_lam = (
            (1 + D_CJ * D_CJ)
            / (gamma + 1.0)
            * (
                1.0
                + (D_CJ * D_CJ - gamma) / (1.0 + D_CJ * D_CJ) * (np.sqrt(1 - y))
            )
        )
        omega = (1 - y) * np.exp(-act_energy / p_lam / V_lam)
        return (u_lam - D_CJ) / omega

    def RHS_lambda(x, y):
        rate_const, _ = integrate.quad(for_reac_rate, 0.0, 0.5)
        rate_const = -rate_const  # ???
        return rate_const / for_reac_rate(y)

    def ZND_init_cond(nodes):
        r = ode(RHS_lambda).set_integrator(
            "zvode", method="bdf"
        )  # using pythonfunction
        r.set_initial_value(0.0, 0.0)
        dx = -(np.abs(nodes[0, 1] - nodes[0, 0])) / 2
        z = []
        lam_x = []
        eps = 1e-6
        cur_lam = 0.0
        node_num = -2
        while (np.abs(cur_lam - 1.0) >= eps) and (node_num >= -nodes.shape[1]):
            z.append(r.t)
            lam_x.append(r.y.real)
            while r.successful() and r.t > nodes[0, node_num]:
                r.integrate(r.t + dx)
            cur_lam = r.y.real
            node_num -= 1
        lam_x = lam_x[::-1]
        lam_x_raw = np.array(lam_x).reshape((1, -1))
        # Here we fill the array for lambda with ones
        # because ODE_integrator does not properly calculate the parts of the domain far from zero
        lam_x = np.hstack(
            (
                np.ones(nodes.shape[1] - lam_x_raw.shape[1]).reshape((1, -1)),
                lam_x_raw,
            )
        )
        D_CJ = np.sqrt(gamma + (gamma * gamma - 1.0) * Q / 2.0) + np.sqrt(
            (gamma * gamma - 1.0) * Q / 2.0
        )
        V_x = (
            gamma
            / D_CJ
            / D_CJ
            * (1 + D_CJ * D_CJ)
            / (gamma + 1.0)
            * (
                1.0
                - (D_CJ * D_CJ - gamma)
                / (1.0 + D_CJ * D_CJ)
                / gamma
                * (np.sqrt(1 - lam_x))
            )
        )
        u_x = (
            1
            / (gamma + 1.0)
            * (D_CJ * D_CJ - gamma)
            / D_CJ
            * (1 + np.sqrt(1 - lam_x))
        )
        p_x = (
            (1 + D_CJ * D_CJ)
            / (gamma + 1.0)
            * (
                1.0
                + (D_CJ * D_CJ - gamma)
                / (1.0 + D_CJ * D_CJ)
                * (np.sqrt(1 - lam_x))
            )
        )
        # Rankine-Hugoniot conditions for variables at x=0
        (rho_a, u_a, p_a, lam_a) = (1.0, 0.0, 1.0, 0.0)
        c_a = np.sqrt(gamma * p_a / rho_a)
        M_a = D_CJ / c_a
        p_s = (
            2.0 * gamma / (gamma + 1.0) * M_a * M_a
            - (gamma - 1.0) / (gamma + 1.0)
        ) * p_a
        rho_s = (
            (gamma + 1.0) * M_a * M_a / (2.0 + (gamma - 1.0) * M_a * M_a)
        ) * rho_a
        u_s = (2.0 / (gamma + 1.0) * (M_a * M_a - 1.0) / (M_a)) * c_a
        #         self.shock_state = np.array(
        #             (rho_s, u_s, p_s, 0.)).reshape((-1, 1))
        #!!!try to do experimental properties

        # rho_new_tr = np.hstack((np.ones(5)*20 ,np.ones(nodes.shape[1]-5)))
        # u_new_tr = np.ones(nodes.shape[1])
        # p_new_tr = np.hstack((np.ones(5)*200 ,np.ones(nodes.shape[1]-5)))
        # p_new_tr = np.hstack((np.ones(5)))
        # lam_new_tr = np.ones(nodes.shape[1])*0
        ZND_array = np.vstack((1 / V_x, u_x, p_x, lam_x))
        # ZND_array = np.vstack((rho_new_tr, u_new_tr,  p_new_tr, lam_new_tr))
        # print(ZND_array)
        return ZND_array

    # ZND solution behind the shock
    eps_ZND = 1e-12
    init_array_non_positive = ZND_init_cond(
        all_nodes[all_nodes <= eps_ZND].reshape(1, -1)
    )

    # Right ghost points
    if NoG:
        ghost_array = init_array_non_positive[:, -1].reshape(-1, 1)
        for i in range(NoG - 1):
            ghost_array = np.hstack(
                (ghost_array, init_array_non_positive[:, -1].reshape(-1, 1))
            )

        init_array = np.hstack((init_array_non_positive, ghost_array))

    else:
        init_array = init_array_non_positive
    # Write data to hdf5 file
    print()
    initial = file.create_dataset(f"E{act_energy}Q{Q}L{L}N{N}", data=init_array)
    rate_const, _ = integrate.quad(for_reac_rate, 0.0, 0.5)
    rate_const = -rate_const  # ???
    initial.attrs["rate_const"] = rate_const
    return rate_const, init_array


# In[16]:


# @nb.njit('f8[:](f8,i4,f8[:,:,:], f8[:], f8[:], f8[:])')
def var_upstream_jit(x, k, table, ambient, wave_numbers, var_upstream_A):
    """
    Function for introducing periodic nonuniformities in variables for the
    upstream state. The first period is a spline that connects the constant
    state with sine periodic state starting from the second period.
    k = 0: 'rho' k = 1: 'u' k = 2: 'p' k = 3: 'lam'
    """
    var_a = ambient[k]
    wave_number = wave_numbers[k]
    upstream_A = var_upstream_A[k]
    if wave_number:
        if (k == 3) or (k == 1):
            # For truly periodic reaction progres
            var_0 = upstream_A
        #                     var_0 = var_a
        else:
            var_0 = var_a
        if x <= 1 / wave_number:
            idx = np.argmin(np.abs(table[k, 0, :] - x))
            value = table[k, 1, idx]
            value_der = table[k, 2, idx]
            return np.array((value, value_der))
        else:
            value = max(
                (0, var_0 + upstream_A * np.sin(2.0 * np.pi * wave_number * x))
            )
            value_der = (
                0
                if value == 0
                else 2.0
                * np.pi
                * wave_number
                * upstream_A
                * np.cos(2.0 * np.pi * wave_number * x)
            )
            return np.array((value, value_der))
    else:
        return np.array((var_a, 0.0))


# @nb.njit('Tuple((f8[:], f8[:]))(f8,b1,f8[:,:,:],f8[:], f8[:], f8[:])')
def calculate_ambient_jit(
    xi, upstream, table, ambient, wave_numbers, var_upstream_A
):
    if not upstream:
        # ambient state
        return ambient, np.array([0.0, 0.0, 0.0, 0.0])
    else:
        # ambient periodic state
        rho_a, rho_der = var_upstream_jit(
            xi, 0, table, ambient, wave_numbers, var_upstream_A
        )
        u_a, u_der = var_upstream_jit(
            xi, 1, table, ambient, wave_numbers, var_upstream_A
        )
        p_a, p_der = var_upstream_jit(
            xi, 2, table, ambient, wave_numbers, var_upstream_A
        )
        lambda_a, lam_der = var_upstream_jit(
            xi, 3, table, ambient, wave_numbers, var_upstream_A
        )
    return np.array((rho_a, u_a, p_a, lambda_a)), np.array(
        (rho_der, u_der, p_der, lam_der)
    )


# In[17]:


class Simulation:
    def __init__(
        self,
        time_limit=None,
        N=101,
        L=10,
        CFL=0.8,
        Runge_Kutta=None,
        act_energy=None,
        heat_release_type=None,
        Q_wave_number=0.0,
        Q_amp=0.0,
        wave_numbers=(0.0, 0.0, 0.0, 0.0),
        var_upstream_A=(0.0, 0.0, 0.0, 0.0),
        store=True,
        c_f=0.0,
        eps=0.0,
        k_for_fric=0.0,
    ):
        """
        NB: WAVE_NUMBERS FOR VELOCITY AND PRESSURE SHOULD BE KEPT ZERO
        FOR THE CURRENT VERSION OF CODE
        Initializes the Simulation object, defines all the parameters
        and checks the necessary stuff, then performs the calculations
        of the state variables inside the comutational domain and gathering
        data for the shock velocity for time interval [0, time_limit].

        Parameters
        ----------
            time_limit        : double
                Final time of simulation
            N, L, CFL         : int, double, double
                Number of nodes, length of the computational mesh,
                and CFL number used during this Simulation
            Runge_Kutta       : int
                Order of the TVD RK
            heat_release_type : str
                Type of the periodic heat release
            Q_wave_number     : double
                Wave number for the heat release oscillations
            Q_amp             : double
                Amplitude for the heat release oscillations
            wave_numbers      : 4, array_like
                Wave numbers for the state variable oscillations
                ahead of the shock
                [0] - for density
                [1] - for velosity
                [2] - for pressure
                [3] - for lambda
            var_upstream_A    : 4, array_like
                Amplitudes for the state variable oscillations
                ahead of the shock. Indexing is the same.
            store             : bool
                Save the results in the hdf5 file or not
        Returns
        -------
            Creates file initials.hdf5 in the current folder
            with the ZND solution for the given parameters.
            If store=True, also creates or works with file
            E={act_energy value}.hdf5 and save the results for
            the detonation speed time evolution in a separate
            dataset defined by L,N,wave_numbers, and var_upstream_A.
        """
        # Naming issues
        method = "WENO5LF"
        if Runge_Kutta:
            self.method = method + "RK" + str(Runge_Kutta)
        else:
            self.method = method
        self.time_limit = time_limit
        self.N = N
        self.CFL = CFL
        self.Runge_Kutta = Runge_Kutta
        self.NoV = 4
        # Names for plot legends
        self.names = ("Density", "Velocity", "Pressure", "Reaction progress")
        self.num_names = (
            "num_Density",
            "num_Velocity",
            "num_Pressure",
            "num_Reaction progress",
        )
        # Calculate only the initial ZND profile
        if not time_limit:
            only_init = True
        else:
            only_init = False

        # Initializing ambient and shock values
        nu = 1.0  # exponent the reaction rate
        gamma = 1.2  # specific heat ratio
        Q = 50.0  # average heat release
        self.shock_speed = []  # to write D values
        self.shock_acc = []  # to write dD/dt values
        self.dist = [0.0]  # to write the shock position in the lab frame
        # Constant ambient state
        self.ambient = np.array([1.0, 0.0, 1.0, 0.0])
        (rho_a, u_a, p_a, lam_a) = self.ambient
        # Chapman-Jouguet velocity
        D_CJ = np.sqrt(gamma + (gamma * gamma - 1.0) * Q / 2.0) + np.sqrt(
            (gamma * gamma - 1.0) * Q / 2.0
        )
        self.shock_speed.append(D_CJ)  # For steady CJ solution
        self.shock_acc.append(0.0)  # For steady CJ solution
        self.time = 0.0
        self.times = [0.0]

        if act_energy:
            self.act_energy = act_energy
            self.file_name = f"E={act_energy}cf{c_f}_article_A={eps}"
            self.group_name = f"L{L}N{N}"
            self.data_name = f"wn{wave_numbers}amp{var_upstream_A}Q_wn{Q_wave_number}Q{Q+Q_amp}cf{c_f}eps{eps}kf{k_for_fric}"
            # self.data_name = f'wn{wave_numbers}amp{var_upstream_A}Q_wn{Q_wave_number}Q{Q+Q_amp}'
            self.attrs = {
                "time_limit": time_limit,
                "final_dist": None,
                "final_time_step": None,
            }

            # Define the mesh and initial conditions
            my_mesh = Mesh1D(-L, 0.0, num_nodes=N, num_ghost=3)
            self.nodes = my_mesh.get_nodes()
            self.step = my_mesh.step
            self.NoG = my_mesh.NoG
            self.lo = my_mesh.beg
            self.hi = my_mesh.end
            #             ones = np.ones((1, self.NoG))

            # Reading or calculating (if needed) the initial ZND solution
            with h5py.File("initials.hdf5", "a") as file:
                if f"E{act_energy}Q{Q}" + self.group_name in file:
                    print("Init cond is already computed")
                    data = file.get(f"E{act_energy}Q{Q}" + self.group_name)
                    self.rate_const = data.attrs["rate_const"]
                    init_array = data[:]
                else:
                    print("Computing init cond")
                    self.rate_const, init_array = calc_initial(
                        file, L, N, self.nodes, self.NoG, gamma, Q, act_energy
                    )

            # Checking if there is a computation with the same parameters
            if (not only_init) and store:
                with h5py.File(self.file_name + ".hdf5", "a") as file:
                    if self.group_name + "/" + self.data_name in file:
                        group = file[self.group_name + "/" + self.data_name]
                        t = group.attrs["time_limit"]
                        if t >= time_limit:
                            raise Exception(f"Already computed till t={t}")
                        else:
                            print(f"Continue computing from t={t}")
                            init_array = group.get("Solution")[:]
                            speed_data = group.get("Shock speed")[:]
                            self.time = speed_data[0, -1]
                            self.times[-1] = self.time
                            self.shock_speed[-1] = speed_data[1, -1]
                            self.dist[-1] = group.attrs["final_dist"]
        else:
            raise AttributeError("Define activation energy")

        # Physical quantities
        wave_numbers = np.array(wave_numbers)
        var_upstream_A = np.array(var_upstream_A)
        if np.sum(wave_numbers):
            self.upstream = True
            self.min_wn = np.min(wave_numbers[wave_numbers > 0.0])
        else:
            self.upstream = False

        def nonuniform_heat(array):
            """
            Calculate the periodic heat release at the points
            defined by the array
            """
            if (
                array > 1 / Q_wave_number
            ).all():  # здесь ты просчитываешь в каждой точке свою периодич функцию.
                return Q * (
                    1.0 + Q_amp * (np.sin(2.0 * np.pi * Q_wave_number * array))
                )
            else:
                heat_array = np.empty_like(array)
                for i, x in enumerate(array[0, :]):
                    if x <= 0.0:
                        heat_array[:, i] = Q
                    else:
                        heat_array[:, i] = Q * (
                            1.0
                            + Q_amp * (np.sin(2.0 * np.pi * Q_wave_number * x))
                        )
                return heat_array

        def heat_release_func(array_in, x=0.0):
            """
            Calculate the periodic heat release
            taking into account the shock wave motion
            Parameters
            ----------
                array_in : array_like
                    Input array of the mesh nodes
                x        : double
                    Current shock position

            Returns
            -------
                Saves data to the attribute of the class object
            """
            if array_in.shape[1] == my_mesh.NoN + 2 * my_mesh.NoG:
                array = array_in
            else:
                array = array_in[
                    :, my_mesh.end - array_in.shape[1] : my_mesh.end
                ]
            if Q_wave_number:
                array_mov = array + x
                heat_array = nonuniform_heat(array_mov)
            else:
                heat_array = Q * np.ones_like(array)
            return heat_array[0, :]

        self.heat_release = heat_release_func(self.nodes.reshape(1, -1))

        def variable_transform(array, toConserved=False):
            """
            Transform an array of physical variables
            to an array of conserved variables and vice versa
            using the optimized version.
            Just renaming for compatibility.
            """
            return variable_transform_jit(
                array, self.heat_release, gamma, toConserved
            )

        my_mesh.set_values(init_array)
        self.init_cond_phys = my_mesh.get_values(domain=True)
        # Turn variables into the conserved form
        my_mesh.set_values(
            variable_transform(my_mesh.get_values(), toConserved=True)
        )
        self.init_cond = my_mesh.get_values()

        def var_upstream(x, k):
            """
            NOW WORKS ONLY WITH DENSITY AND LAMBDA VARIATIONS
            Function for introducing periodic nonuniformities in variables for the
            upstream state. The first period is a spline that connects the constant
            state with sine periodic state starting from the second period.
            k = 0: 'rho', k = 1: 'u', k = 2: 'p', k = 3: 'lam'.
            """
            var_a = self.ambient[k]
            wave_number = wave_numbers[k]
            upstream_A = var_upstream_A[k]
            if wave_number:
                if k == 0:
                    # Periodic density from TEMPERATURE disturbances
                    # Construction of the spline for the first period
                    if x <= 1 / wave_number:
                        x_for_const = np.linspace(0.0, 1.0 / wave_number / 2.0)
                        y_for_const = var_a * np.ones_like(x_for_const)
                        x_for_periodic = np.linspace(
                            1.0 / wave_number, 2.0 / wave_number
                        )
                        #                     y_for_periodic = np.array(
                        #                         var_0+upstream_A*np.sin(2.*np.pi*wave_number*x_for_periodic))
                        y_for_periodic = np.array(
                            1.0
                            / (
                                1.0
                                + upstream_A
                                * (
                                    np.cos(
                                        2.0
                                        * np.pi
                                        * wave_number
                                        * x_for_periodic
                                    )
                                    - 1.0
                                )
                            )
                        )
                        y_for_periodic = np.where(
                            y_for_periodic >= 0, y_for_periodic, 0
                        )
                        x_for_spline = np.hstack([x_for_const, x_for_periodic])
                        y_for_spline = np.hstack([y_for_const, y_for_periodic])
                        tck = interpolate.splrep(x_for_spline, y_for_spline)
                        value = max((0, interpolate.splev(x, tck, der=0)))
                        value_der = (
                            0
                            if value == 0
                            else interpolate.splev(x, tck, der=1)
                        )
                        return np.array((value, value_der))
                    else:
                        #                     value = max((0, var_0+upstream_A*np.sin(2.*np.pi*wave_number*x)))
                        #                     value_der = 0 if value==0 else 2.*np.pi*wave_number*upstream_A*np.cos(2.*np.pi*wave_number*x)
                        # For temperature
                        value = max(
                            (
                                0,
                                1.0
                                / (
                                    1.0
                                    + upstream_A
                                    * (
                                        np.cos(2.0 * np.pi * wave_number * x)
                                        - 1.0
                                    )
                                ),
                            )
                        )
                        value_der = (
                            0
                            if value == 0
                            else 2.0
                            * np.pi
                            * wave_number
                            * upstream_A
                            * np.sin(2.0 * np.pi * wave_number * x)
                            * value**2
                        )
                        return np.array((value, value_der))
                if k == 3:
                    # Periodic lambda from fresh mixture
                    # Construction of the spline for the first period
                    if x <= 1 / wave_number:
                        x_for_const = np.linspace(0.0, 1.0 / wave_number / 2.0)
                        y_for_const = var_a * np.ones_like(x_for_const)
                        x_for_periodic = np.linspace(
                            1.0 / wave_number, 2.0 / wave_number
                        )
                        #                     y_for_periodic = np.array(
                        #                         var_0+upstream_A*np.sin(2.*np.pi*wave_number*x_for_periodic))
                        y_for_periodic = np.array(
                            1.0
                            - np.exp(
                                upstream_A
                                * (
                                    np.cos(
                                        2 * np.pi * wave_number * x_for_periodic
                                    )
                                    - 1
                                )
                            )
                        )
                        y_for_periodic = np.where(
                            y_for_periodic >= 0, y_for_periodic, 0
                        )
                        x_for_spline = np.hstack([x_for_const, x_for_periodic])
                        y_for_spline = np.hstack([y_for_const, y_for_periodic])
                        tck = interpolate.splrep(x_for_spline, y_for_spline)
                        value = max((0, interpolate.splev(x, tck, der=0)))
                        value_der = (
                            0
                            if value == 0
                            else interpolate.splev(x, tck, der=1)
                        )
                        return np.array((value, value_der))
                    else:
                        #                     value = max((0, var_0+upstream_A*np.sin(2.*np.pi*wave_number*x)))
                        #                     value_der = 0 if value==0 else 2.*np.pi*wave_number*upstream_A*np.cos(2.*np.pi*wave_number*x)
                        # For temperature
                        value = max(
                            (
                                0,
                                1.0
                                - np.exp(
                                    upstream_A
                                    * (np.cos(2 * np.pi * wave_number * x) - 1)
                                ),
                            )
                        )
                        value_der = (
                            0
                            if value == 0
                            else 2.0
                            * np.pi
                            * wave_number
                            * upstream_A
                            * np.sin(2.0 * np.pi * wave_number * x)
                            * (1.0 - value)
                        )
                        return np.array((value, value_der))
            else:
                return np.array((var_a, 0.0))

        def calculate_ambient(xi, upstream):
            """
            Combine together all the state variables
            ahead of the shock for uniform and non-uniform
            media
            """
            Temp = 1.0  # For uniform temperature
            if not upstream:
                # ambient state
                return np.array([1.0, 0.0, 1.0, 0.0]), np.array(
                    [0.0, 0.0, 0.0, 0.0]
                )
            else:
                # ambient periodic state
                rho_a, rho_der = var_upstream(xi, 0)
                u_a, u_der = var_upstream(xi, 1)
                p_a, p_der = var_upstream(xi, 2)
                lambda_a, lam_der = var_upstream(xi, 3)
                return np.array((rho_a, u_a, p_a, lambda_a)), np.array(
                    (rho_der, u_der, p_der, lam_der)
                )

        # The main loop
        if not only_init:
            beg = self.lo
            end = self.hi
            start = beg - 1
            finish = end - 1
            solution = self.init_cond
            cur_sec = 0
            # check = []
            # full_solution=[]
            # full_solution.append(init_array)

            # displacement = 0
            # t = tqdm.tqdm(range(int(time_limit)), desc='Time integration')
            while self.time < time_limit:
                D_cur = self.shock_speed[-1]
                xi = self.dist[-1]
                ambient, ambient_der = calculate_ambient(xi, self.upstream)
                # RK 3-5 begins
                solution, D_next, self.tau, c_fric, D_acc = RK(
                    Runge_Kutta,
                    solution,
                    self.heat_release,
                    ambient,
                    ambient_der,
                    self.step,
                    D_cur,
                    gamma,
                    self.act_energy,
                    self.rate_const,
                    self.CFL,
                    start,
                    finish,
                    beg,
                    end,
                    self.nodes.reshape(1, -1)[0],
                    xi,
                    c_f,
                    eps,
                    k_for_fric,
                )
                if self.time + self.tau < time_limit:
                    self.time += self.tau
                else:
                    self.tau = time_limit - self.time
                    self.time = time_limit

                self.shock_speed.append(D_next)
                self.shock_acc.append(D_acc)
                self.times.append(self.times[-1] + self.tau)
                self.dist.append(self.dist[-1] + self.tau * D_cur)
                self.heat_release = heat_release_func(
                    self.nodes.reshape(1, -1), x=self.dist[-1]
                )
                if self.time > cur_sec:
                    cur_sec += 1
                    # t.update()
                # print(c_fric.shape)
            # check.append(c_fric)

            # print(check)
            # t.close()
            results = np.vstack(
                (
                    np.array(self.times),
                    np.array(self.shock_speed),
                    np.array(self.shock_acc),
                )
            )
            self.attrs["final_dist"] = self.dist[-1]
            self.attrs["final_time_step"] = self.tau
            my_mesh.set_values(solution)

        # self.check = check
        # self.sol = full_solution

        # Turn the variables into the physical form
        conserved_solution = my_mesh.get_values()
        my_mesh.set_values(
            variable_transform(conserved_solution, toConserved=False)
        )
        self.num_solution = my_mesh
        physical_solution = my_mesh.get_values()

        # Write results into file
        if (not only_init) and store:
            with h5py.File(self.file_name + ".hdf5", "a") as file:
                if self.group_name + "/" + self.data_name in file:
                    print("Adding data to datasets..")
                    group = file[self.group_name + "/" + self.data_name]
                    group.attrs.modify("time_limit", self.attrs["time_limit"])
                    group.attrs.modify("final_dist", self.attrs["final_dist"])
                    group.attrs.modify(
                        "final_time_step", self.attrs["final_time_step"]
                    )
                    dset = group.get("Shock speed")
                    old_len = dset.shape[1]
                    new_len = old_len + len(self.times)
                    dset.resize(new_len, axis=1)
                    dset[0, old_len:] = results[0, :]
                    dset[1, old_len:] = results[1, :]
                    dset[2, old_len:] = results[2, :]
                    # group['Solution'][:] = physical_solution
                else:
                    print("Creating new datasets..")
                    if (
                        np.isnan(results).any()
                        or np.isnan(physical_solution).any()
                    ):
                        print("There are NaNs in the solution. No data saved")
                    else:
                        group = file.create_group(
                            self.group_name + "/" + self.data_name
                        )
                        group.attrs.update(self.attrs)
                        dset = group.create_dataset(
                            "Shock speed",
                            data=results,
                            maxshape=(results.shape[0], None),
                            compression="lzf",
                            fletcher32=True,
                        )
                        # dset = group.create_dataset(
                        # 'Solution', data = physical_solution)#, compression='lzf', fletcher32=True)
                        # dset = group.create_dataset(
                        #        'Solution_full', data = full_solution)#, compression='lzf', fletcher32=True)


if __name__ == "__main__":
    # Catching command line arguments using argparse
    import argparse

    # Instantiate the parser
    parser = argparse.ArgumentParser(description="DetWENORK3")
    # Required positional arguments
    parser.add_argument(
        "act_energy",
        type=float,
        help="A required float activation energy argument",
    )
    parser.add_argument(
        "dom_length", type=float, help="A required float domain length argument"
    )
    parser.add_argument(
        "num_nodes", type=int, help="A required int number of nodes argument"
    )
    #   parser.add_argument('amp', type=float,
    #                       help='A required amplitude of the periodicity')
    #   parser.add_argument('amp', type=float,
    #                       help='A required amplitude of the periodicity')
    parser.add_argument(
        "c_f", type=float, help="A required float coefficient of friction"
    )
    parser.add_argument(
        "eps",
        type=float,
        help="A required eps (coefficient in front of the sine )",
    )
    parser.add_argument(
        "k_for_fric_start",
        type=float,
        help="A required starting period of friction",
    )
    parser.add_argument(
        "k_for_fric_end",
        type=float,
        help="A required ending period of friction",
    )
    parser.add_argument(
        "k_for_fric_step",
        type=float,
        help="A required step for period of friction",
    )
    args = parser.parse_args()
    print(
        f"Run simulation from k_for_fr={args.k_for_fric_start} to k_for_fr={args.k_for_fric_end} with the step {args.k_for_fric_step}"
    )

    k_for_frics = np.round(
        np.arange(
            args.k_for_fric_start,
            args.k_for_fric_end + args.k_for_fric_step / 2.0,
            args.k_for_fric_step,
        ),
        6,
    )

    # For conditions into RDE
    for k_for_fr in k_for_frics:
        print(f"k_for_fric={k_for_fr}")
        try:
            # Simulation(act_energy=args.act_energy, Runge_Kutta=3,
            #         L = args.dom_length, N=args.num_nodes, time_limit=3000,
            #        wave_numbers=(0.0,0.0,0.0,0),var_upstream_A=(0.0,0.0,0.0,0.0),
            #        c_f = args.c_f, eps = args.eps, k_for_fric = k_for_fr, CFL=0.8)
            amp = args.eps
            Simulation(
                act_energy=args.act_energy,
                Runge_Kutta=3,
                L=args.dom_length,
                N=args.num_nodes,
                time_limit=2000,
                wave_numbers=(0.0, 0.0, 0.0, 0.0),
                var_upstream_A=(0.0, 0.0, 0.0, 0.0),
                c_f=args.c_f,
                eps=amp,
                k_for_fric=k_for_fr,
                CFL=0.8,
            )
        except Exception as e:
            print(e)


# In[ ]:
