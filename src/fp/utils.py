import numba as nb

@nb.vectorize("f8(f8)")
def mexp_jit(x):
    """
    Numba optimized exponent
    """
    return math.exp(x)

# Separate into 2 functions without flag
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

