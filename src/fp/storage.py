# Organize everything here

            self.file_name = f"E={act_energy}cf{c_f}_article_A={eps}"
            self.group_name = f"L{L}N{N}"
            self.data_name = f"wn{wave_numbers}amp{var_upstream_A}Q_wn{Q_wave_number}Q{Q+Q_amp}cf{c_f}eps{eps}kf{k_for_fric}"
            # self.data_name = f'wn{wave_numbers}amp{var_upstream_A}Q_wn{Q_wave_number}Q{Q+Q_amp}'
            self.attrs = {
                "time_limit": time_limit,
                "final_dist": None,
                "final_time_step": None,
            }

# Write results into hdf5 file
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
