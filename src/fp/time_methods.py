import numba as nb

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

