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


def main(job_names, command_strs):
    for command_str, job_name in zip(command_strs, job_names):
        shutil.copyfile(file_name, submit_name)
        with fileinput.FileInput(submit_name, inplace=True) as file:
            for line in file:
                print(line.replace(replace_args, command_str), end="")

        with fileinput.FileInput(submit_name, inplace=True) as file:
            for line in file:
                print(line.replace(replace_name, job_name), end="")
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
        "cma_opts": {"popsize": 100, "tolfun": 1e-3, "tolx": 1e-3},
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
    }

    dir_name = "Experiments/ManyTrials"

    N_trails = 50

    job_names = list()
    arg_paths = list()
    for trial in range(N_trails):

        args["save_path"] = dir_name + f"/trial_{trial}"
        if args["save_path"] is not None:
            save_path = Path(args["save_path"])
            save_path.mkdir(parents=True, exist_ok=True)

            with open(save_path / "params.yaml", "w") as yaml_file:
                yaml.dump(args, yaml_file, default_flow_style=False)

        job_names.append(f"train_{trial}")
        arg_paths.append(str(save_path / "params.yaml"))

    main(job_names, arg_paths)
