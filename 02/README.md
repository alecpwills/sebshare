This directory contains the evaluation/training outputs from training the *pre-trained SCAN model* with *default `train.py` inputs* on the test training subset:
- H2
- H

Plotted graphs show evaluation of the models on the validation set:
#### Validation AE
- S2
- C2H2
- BeH
- NO2
- CH4
- PF3
- CH2
- CO2HCH3
#### Validation BH
- H + N2O -> HON2
- OH + CL -> CLHO

- `02_valloss_last.png` contains plots showing the evaluated network's predicted AE and total energies, compared to reference (literature or CCSD(T)) and XCDiff. The last generated model's evaluation here is plotted.
- `02_valloss_epochs.png` contains plots showing the evaluated AE and total energies as the epochs increase.
- `02_valloss_rho.png` contains a plot showing the validation dataset index vs. the density loss for each model.
- `02_valloss_bh1.png` contains a plot of the deviation in predicted barrier heights for the H+N2O -> HON2 reaction, along with the deviation from XCDiff's prediction
- `02_valloss_bh2.png` contains a plot of the deviation in predicted barrier heights for the OH+Cl -> HClO reaction, along with the deviation from XCDiff's prediction
- `MODEL_MGGA_e*/` are the evaluation directories -- eXX is the XXth model generated during training.
    - `MODEL_MGGA_e00` is just a copy of the pre-trained SCAN network before evaluation.
    - Each directory contains that model's enhancement factors plotted with $\alpha$ = 0, 1, or 10
        - `_alpha*.pdf` contains the plot for each given alpha
        - `_alpha*_x.npy` contains the plotted x-axis values
        - `_alpha*_y.npy` contains the plotted x-axis values
    - Each directory further contains the `val` subdirectory, wherein the model is used to predict validation energies
- `train` has the training outputs from the training process.
- `trainloss_*` figures plot the:
    - molecule-wise atomization energy losses as training progresses (`trainloss_ae_*`)
    - atom-wise total energy losses and molecule-wise density losses (`trainloss_ee_*`)
    - *SCALED* aggregate losses for AE, rho, TE, as training progresses (`trainloss_te_*`)
        - Scaled by the weights given to the losses during training.
        - Files with `_dev` in the name plot the total deviation from the first epoch's value
        - Files with `_per` in the name plot the percent change from the first epoch's value
        - Files with `_log` in the name plot the values on a symlog y-axis
- `bins` contains the histograms of each training molecule's binned $x_1$, $x_2$, and $x_3$ parameters.
    - `bin.pdf` contains the total binned histogram, combining each molecule in the training set.
    - `bin_zoom.pdf` contains the same image as above, but with a lower x-limit of -20.
    - `{idx}_{symbols}_bin.pdf` contain the molecule's individual parameter binning