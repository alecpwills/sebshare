Directory descriptions:

- `xcd` contains the output of evaluating the published xcdiff network on the validation set
- `01` contains the training output and outputs of check points from training the pre-trained SCAN model on a subset of XCDiff training molecules evaluated on the validation set.
    - `train` subdirectory contains the `.config` file output by `train.py` capturing the arguments passed to the program
- `data` contains subdirectories containing the CCSD(T) calculations of the training subset (`test_subset_ps`, file called `subat_refres.traj`) and the validation set (`validation`, file called `val_c.traj`)
- `scripts` contains the relevant dpyscf-lite scripts (`prep_data.py`, `train.py`, `eval.py`) and the script used to generate the CCSD(T) reference data (`run_pyscf.py`)
- `notebooks` contains
