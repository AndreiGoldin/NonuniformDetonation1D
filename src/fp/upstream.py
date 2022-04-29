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
                y_for_periodic = np.array( 1.0 / ( 1.0 + upstream_A * ( np.cos( 2.0 * np.pi * wave_number * x_for_periodic) - 1.0)))
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
                value = max( ( 0, 1.0 / ( 1.0 + upstream_A * ( np.cos(2.0 * np.pi * wave_number * x) - 1.0)),))
                value_der = ( 0 if value == 0 else 2.0 * np.pi * wave_number * upstream_A * np.sin(2.0 * np.pi * wave_number * x) * value**2)
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
                y_for_periodic = np.array( 1.0 - np.exp( upstream_A * ( np.cos( 2 * np.pi * wave_number * x_for_periodic)-1)))
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
                value = max( ( 0, 1.0 - np.exp( upstream_A * (np.cos(2 * np.pi * wave_number * x) - 1)),))
                value_der = ( 0 if value == 0 else 2.0 * np.pi *
                        wave_number * upstream_A * np.sin(2.0 * np.pi *
                            wave_number * x) * (1.0 - value))
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
        (rho_der, u_der, p_der, lam_der))

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
