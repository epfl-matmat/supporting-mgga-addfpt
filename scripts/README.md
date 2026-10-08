## Setup
1. Install Julia v1.12.5. The scripts in this repository assume that juliaup is installed. This can be tested with `juliaup --version`. To use Julia v1.12.5 you can then:
    1. Install the `1.12.5` version channel with `juliaup add 1.12.5`.
    2. Run Julia v1.12.5 with the `julia +1.12.5` command.
2. From this folder, start julia with `julia +1.12.5 --project`.
3. Install the required dependencies from the Julia shell with `using Pkg; Pkg.instantiate()`.
4. Some scripts use the `~/.julia/bin/mpiexecjl` executable, which can be installed with `using MPI; MPI.install_mpiexecjl()`.

## General remarks
- Running a script named `script.jl` is done by running `julia +1.12.5 --project scripts.jl`. Additional arguments can be specified after the filename.
- The scripts involving a more expensive calculation can generally be ran with `bash script.jl <additional arguments>`, which will take care of invoking julia. See the "shebang" at the beginning of the script. Depending on the computer, the `-n` (mpiexec process count), `-t` (Julia thread count) and `--heap-size-hint` (Julia target max RAM) arguments might need to be adjusted.
- The scripts will likely run on Julia versions newer than 1.12.5. In that case, the `+1.12.5` can be removed or replaced by another release channel.

## Folder contents
### Elastic properties of diamond
The `results1_elastic_constants` folder contains the calculation of the elastic constants of diamond.

- See the `2_elastic_constants_addfpt.jl` script for the calculation with AD-DFPT. Note that it takes an extra `C` argument that identifies the structure.
- See the `3_elastic_constants_findiff.jl` script for the calculation with finite differences, also taking an extra `C` argument.
- See the `4_plot_data.jl` script to generate Figure 1 of the paper.

### Response properties of ZnO and BaTiO₃
The `results2_response_properties` folder contains the calculation of the response properties of ZnO and BaTiO₃. Each calculation starts with a relaxation (`1_relax.jl`), followed by an SCF (`2_scf.jl`), followed by the response computations (`3_ddk.jl` and `4*_perturb_*.jl`), followed by an analysis (`5_analyze.jl`).

- The following files are shared across all the calculations:
  - The `3_ddk.jl` file for the calculation of $\frac{\mathrm{d}u_{n\mathbf{k}}}{\mathrm{d}k_\alpha}$.
  - The `4a_perturb_elfield.jl`, `4b_perturb_strain.jl` and `4c_perturb_atom.jl` files for the response calculation to different types of perturbations.
  - The `5_analyze.jl` file that postprocesses the results of these calculations and produces the data shown in the tables of the paper.
- For each crystal and XC functional, there is a corresponding folder, e.g. `ZnO_lda`, containing:
  - The starting and relaxed structures in `structures/`.
  - The used parameters `0_params.jl` and the relaxation (`1_relax.jl`) and SCF (`2_scf.jl`) scripts.
  - A `run_postprocess.sh` script that runs all the DFPT computations after the SCF has been performed.
  - (for r2SCAN01) The used pseudopotentials in `pseudos/`.
  - The raw DFPT output tensors as JLD2 files in `outputs/`.
  - The logs of all scripts, including notably `5_analyze.log` which contains all the results presented in the paper.
- To rerun the analysis, e.g. for `ZnO_lda`, go to the corresponding folder and run
    ```
    bash ../5_analyze.jl ZnO
    ```
    For the analysis of a BaTiO₃ calculation, replace `ZnO` by `BaTiO3`.
- To regenerate the DFPT results, e.g. for `ZnO_lda`, go to the corresponding folder and run
    ```
    bash 2_scf.jl
    bash run_postprocess.jl
    ```
    This will take several hours.

### Density-based XC functional optimization
The `results3_xc_optimization` folder contains the reference data and scripts used to optimize an exchange meta-GGA functional for silicon.

- The `dataset_si_k8q8_v0_jld2` folder contains the HSE06 and PBE reference data computed with Quantum ESPRESSO.
  - There is one JLD2 file per functional (`hse06.jld2` and `pbe.jld2`), each containing 35 calculation results: 5 considered phases (including diamond and β-tin) at 7 volumes.
  - The Quantum ESPRESSO input and output files are also stored verbatim in these JLD2 files.
  - See also [the corresponding README](dataset_si_k8q8_v0_jld2/README.md) and the [example loading script](dataset_si_k8q8_v0_jld2/example_load.jl).
- The `train.sh` is the central script used to train the meta-GGA exchange functional.
  - The XC functional is defined in `xc_model.jl`.
  - The loss function evaluation and training loop is in `train.jl`.
  - For each of the tree fitting strategies, there is a `config_*.jl` file.
    For each of them, there is a corresponding `runs/config_*/` folder containing the training log
    and the final training snapshot in a JLD2 file.
- The `evaluate.jl` script is used to compute the EOS after training for evaluation. It will take a training snapshot `runs/config_*/bfgs_<id>.jld2` and produce a corresponding evaluation data file `runs/config_*/eval/bfgs_<id>.jld2`.
- The `figure.jl` script generates Figure 2 of the paper. To regenerate the figure, run the following command from the same folder:
  ```jl
  julia +1.12.5 --project figure.jl runs/config_E_20260924-105331/ runs/config_Erho_20260922-201051/ runs/config_Erhotau_20260922-201052/
  ```
  This produces `figure_lossvariants.pdf` and the extra `figure_lossvariants.json` data file.

### Pseudopotential verification
The `pseudo_verification` folder contains the verification of the METAPSP-generated pseudopotentials against reference all-electron (AE) data, as presented in the supplementary.

- See the subfolders (one per element) for details of the equation of state calculations.
- See the `compare_eos.jl` script for the comparison between the plane-wave results and the AE reference. The script cannot be ran without the AE reference data, however the log file `compare_eos.log` contains all the results presented in the supplementary.
