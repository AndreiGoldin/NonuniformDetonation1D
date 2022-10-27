# Contains everything to read input from different sources
import yaml

class Reader:
    def __init__(self, input_format):
        self.supported_formats = ['yaml', 'YAML']
        self.supported_problems = ["Advection", "Burgers", "Euler", "ReactiveEuler"]
        if input_format in self.supported_formats:
            self.format = input_format
        else:
            raise NotImplementedError(f"Supported input formats are {*self.supported_formats,}")

    def get_input(self, filename):
        if self.format in ['yaml', 'YAML']:
            with open(filename , "r") as stream:
                try:
                    loaded_p = yaml.safe_load(stream)
                except yaml.YAMLError as exc:
                    print(exc)
            if loaded_p["problem"]["type"] not in self.supported_problems:
                raise NotImplementedError(f"Supported problems are {*self.supported_problems,}")
            parameters = {}
            parameters["a"] = loaded_p["domain"]["left boundary"]
            parameters["b"] = loaded_p["domain"]["right boundary"]
            parameters["N"] = loaded_p["domain"]["nodes"]
            assert parameters["N"] > 0, "The number of nodes should be positive."
            parameters["T"] = loaded_p["domain"]["timespan"]
            if "timesteps" in loaded_p["domain"]:
                parameters["Nt"] = loaded_p["domain"][ "timesteps"]
                assert parameters["Nt"] > 0, "The number of timesteps should be positive."
            assert parameters["T"] >= 0, "The final time should be non-negative."
            if loaded_p["problem"]["frame"] == "Laboratory":
                parameters["frame"] = "LFOR"
            elif loaded_p["problem"]["frame"] == "Shock":
                parameters["frame"] = "SAFOR"
                if parameters["b"] != 0.0:
                    print("The shock must be located at x=0. Changing right boundary to zero...")
                    parameters["b"] = 0.0
            else:
                raise AttributeError("Only Laboratory or Shock frame are allowed.")
            assert parameters["a"] < parameters["b"], "The left boundary should be less than the right boundary."
            parameters["solver_type"] = loaded_p["problem"]["type"]
            if "ReactiveEuler" in parameters["solver_type"]:
                assert parameters["a"] < 0.0, "The shock must be located at x=0. Please, choose negative left boundary."
            parameters["init_cond_type"] = loaded_p["problem"]["initial"]
            parameters["bound_cond_type"] = loaded_p["problem"]["boundary"]
            parameters["upstream_cond_type"] = loaded_p["upstream"]["type"]
            parameters["space_method"] = loaded_p["methods"]["space"]
            parameters["time_method"] = loaded_p["methods"]["time"]
            if parameters["frame"] == "SAFOR":
                parameters["solver_type"] += "SAFOR"
                parameters["time_method"] += "_SAFOR"
                parameters["bound_cond_type"] += "_SAFOR"
            problem_p = loaded_p["problem"]["parameters"]
            upstream_p = loaded_p["upstream"]["parameters"]
            if not problem_p:
                problem_p = {}
            if not upstream_p:
                 upstream_p = {}
            parameters = {**parameters, **problem_p, **upstream_p}
            # parameters = {**parameters, **loaded_p["problem"]["parameters"], **loaded_p["upstream"]["parameters"]}
            if "activation energy" in parameters:
                parameters["act_energy"] = parameters.pop("activation energy")
            if "heat release" in parameters:
                parameters["heat_release"] = parameters.pop("heat release")
            if "density amplitude" in parameters:
                parameters["Arho"] = parameters.pop("density amplitude")
            if "density wavenumber" in parameters:
                parameters["krho"] = parameters.pop("density wavenumber")
            if "lambda amplitude" in parameters:
                parameters["Alam"] = parameters.pop("lambda amplitude")
            if "lambda wavenumber" in parameters:
                parameters["klam"] = parameters.pop("lambda wavenumber")

            callbacks = loaded_p["callbacks"]
            save_tag = ""
            param_tag = ""
            for (key,value) in {**problem_p, **upstream_p}.items():
                key = "_".join(map(lambda x: x[:3], key.split(" ")))
                param_tag += f"{key}={value:.2f}, "
                save_tag += f"_{key}={value:.2f}"

            return parameters, callbacks, param_tag[:-2], save_tag
