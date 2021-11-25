import numpy as np
from pathlib import Path
import yaml
import os
import shutil
import fileinput

from src.utils import A_0, nm

file_name = "sbatch.template"
submit_name = "sbatch.sbatch"
replace_name = "JOB_NAME"
replace_args = "ARGS_PATH"
replace_path = "DIR_PATH"


def main(job_names, command_strs):
    for command_str, job_name in zip(command_strs, job_names):
        shutil.copyfile(file_name, submit_name)
        with fileinput.FileInput(submit_name, inplace=True) as file:
            for line in file:
                print(line.replace(replace_args, command_str), end="")

        with fileinput.FileInput(submit_name, inplace=True) as file:
            for line in file:
                print(line.replace(replace_name, job_name), end="")

        with fileinput.FileInput(submit_name, inplace=True) as file:
            dir_path = str(Path(command_str).parent)
            for line in file:
                print(line.replace(replace_path, dir_path), end="")

        os.system(f"sbatch {submit_name}")

    print("DONE")


if __name__ == "__main__":
    args = {
        "target_angle": np.pi / 2.0,
        "target_dist": A_0 * nm,
        "w_dist_orient": 0.5,
        "w_reg": 500.0,
        "w_phase": 0.0,
        "cma_sigma": 0.25,
        "cma_opts": {"popsize": 100, "tolfun": 1e-3, "tolx": 5e-4},
        "ZeroCoeffs": [0, 1, 2, 5, 6, 7, 8, 9, 10],
        "x_0_init": 0.25,
        "width": 400 * nm,
        "polarization": [1, 1j],
        "Nmax": 4,
        "rho_scale": 1.0,
        "power": 0.25,
        "radius": 75 * nm,
        "lmax": 2,
        "wavelength": 770 * nm,
        "e_field_sampling": 50,
        "min_sep_dist": 200 * nm,
        "max_sep_fac": 2,
        "save_path": None,
        "save_freq": 10,
        "sim_init_sep": 700 * nm,
        "sim_init_angle": np.pi / 2.0,
    }

    # dir_name = "Experiments/Trials_50"
    # N_trails = 50

    dir_name = "Experiments/Trials_sep_params"
    N_trails = 10
    sep_dists = [150 * nm, 200 * nm, 250 * nm]
    max_sep_facs = [2, 3, 4]
    x_0_inits = [0.25, [0.8239384, 0.02414681, -3.40704586, 1.37237894, -0.14930837]]
    w_regs = [100.0, 500.0, 2500.0]

    job_names = list()
    arg_paths = list()
    for trial in range(N_trails):
        for sep_dist in sep_dists:
            for max_sep_fac in max_sep_facs:
                for x_0_init in x_0_inits:
                    for w_reg in w_regs:
                        if isinstance(x_0_init, list):
                            x_0_name = "fixed"
                        else:
                            x_0_name = "random"
                        exp_name = f"/sep_dist={int(sep_dist / nm)}-max_sep_fac={max_sep_fac}-x_0={x_0_name}-w_reg={int(w_reg)}-trial={trial}"

                        args["sep_dist"] = sep_dist
                        args["max_sep_fac"] = max_sep_fac
                        args["x_0_init"] = x_0_init
                        args["w_reg"] = w_reg

                        args["save_path"] = dir_name + exp_name
                        if args["save_path"] is not None:
                            save_path = Path(args["save_path"])
                            save_path.mkdir(parents=True, exist_ok=True)

                            with open(save_path / "params.yaml", "w") as yaml_file:
                                yaml.dump(args, yaml_file, default_flow_style=False)

                        job_names.append(exp_name)
                        arg_paths.append(str(save_path / "params.yaml"))

    main(job_names, arg_paths)
