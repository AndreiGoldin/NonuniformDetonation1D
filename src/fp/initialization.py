# Write get_input()
# Maybe use abstract class Parameters and subclasses DetParams, RDEParams, FricParams, SingleParams
# with methods for getting and checking parameters
def get_input():
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
        f"Run simulation from k_for_fr={args.k_for_fric_start} to
        k_for_fr={args.k_for_fric_end} with the step {args.k_for_fric_step}"
    )

    k_for_frics = np.round(
        np.arange(
            args.k_for_fric_start,
            args.k_for_fric_end + args.k_for_fric_step / 2.0,
            args.k_for_fric_step,
        ),
        6,
    )
    wave_numbers = np.array(wave_numbers)
    var_upstream_A = np.array(var_upstream_A)
    if np.sum(wave_numbers):
        self.upstream = True
        self.min_wn = np.min(wave_numbers[wave_numbers > 0.0])
    else:
        self.upstream = False

def allocate_lists():
######### Check that the necessary parameters are difined
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
    self.act_energy = act_energy

def create_mesh():
    my_mesh = Mesh1D(-L, 0.0, num_nodes=N, num_ghost=3)
    self.nodes = my_mesh.get_nodes()
    self.step = my_mesh.step
    self.NoG = my_mesh.NoG
    self.lo = my_mesh.beg
    self.hi = my_mesh.end
    #             ones = np.ones((1, self.NoG))

def set_initial():
    my_mesh.set_values(init_array)
    self.init_cond_phys = my_mesh.get_values(domain=True)
    # Turn variables into the conserved form
    my_mesh.set_values(variable_transform(my_mesh.get_values(), toConserved=True))
    self.init_cond = my_mesh.get_values()
