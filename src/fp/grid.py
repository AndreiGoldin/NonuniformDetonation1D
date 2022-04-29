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

