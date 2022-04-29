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

# Check already computed
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

def solve(functions):
    # Unpack list of functions
    timer = start_timer()

    params = get_input()
    already_computed, resume_computations = check_params(params)

    if already_computed:
        raise Exception(f"Already computed for {params}")

    mesh = create_mesh()
    if resume_computations:
        state = read_state()
    else:
        state = set_initial(mesh)

    # Tell that computation starts

    while t<T:
        check_computations()
        state, dt = timeintegrate(state)
        t += dt

    post_process(state) # find spectra, peaks, etc
    write_solution(state) # or anything else
    timer_stop()

    # Summary


if __name__ == '__main__':
    problem_type = 'RDE'
    problem_functions = choose_functions(problem_type)
    solve(problem_functions)
