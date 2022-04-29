import numba as nb

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
