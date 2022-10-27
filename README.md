Directory descriptions:

- `xcd` contains the output of evaluating the published xcdiff network on the validation set
    - the `nan` values for the atomization energies can be ignored for now -- there is a conflict with how I store the barrier height data that yields nans for the final evaluation, but the scripts within the `notebooks` directory generate the MAE for the AE correctly.
- `01` contains the training output and outputs of check points from training the pre-trained SCAN model on a subset of XCDiff training molecules evaluated on the validation set. These models were generated with the default network parameters/weights.
    - `01` contains README further clarifying the test training subset and the validation set.
    - `train` subdirectory contains the `.config` file output by `train.py` capturing the arguments passed to the program
- Directories `02-07` contain training runs with progressively larger subsets of training data
    - `02` contains: H2, H
    - `03` contains: H2, N2, H, N
    - `04` contains: H2, N2, LiF, F, H, Li, N
    - `05` contains: H2, N2, LiF, CNH, C, F, H, Li, N
    - `06` contains: H2, N2, LiF, CNH, CO2, C, F, H, Li, N, O
    - `07` contains: H2, N2, LiF, CNH, CO2, Cl2, C, Cl, F, H, Li, N, O
- `data` contains subdirectories containing the CCSD(T) calculations of the training subset (`test_subset_ps`, file called `subat_refres.traj`) and the validation set (`validation`, file called `val_c.traj`)
- `scripts` contains the relevant dpyscf-lite scripts (`prep_data.py`, `train.py`, `eval.py`) and the script used to generate the CCSD(T) reference data (`run_pyscf.py`)
- `notebooks` contains the scripts to generate the current figures
