This directory contains the evaluation/training outputs from training the *pre-trained SCAN model* with *default `train.py` inputs* on the full published training set:
#### 10 Linear Closed-Shell Molecules
- $\mathrm{H_2}$
- $\mathrm{N_2}$
- $\mathrm{LiF}$
- $\mathrm{CNH}$
- $\mathrm{CO_2}$
- $\mathrm{F_2}$
- $\mathrm{C_2H_2}$
- $\mathrm{OC}$
- $\mathrm{LiH}$
- $\mathrm{Na_2}$

#### 3 Linear Open-Shell Molecules
- $\mathrm{NO}$
- $\mathrm{CH}$
- $\mathrm{OH}$

#### 8 Non-Linear Closed-Shell Molecules
- $\mathrm{NO_2}$
- $\mathrm{NH}$
- $\mathrm{O_3}$
- $\mathrm{N_2O}$
- $\mathrm{CH_3}$
- $\mathrm{CH_2}$
- $\mathrm{H_2O}$
- $\mathrm{NH_3}$

*A set of the constituent atoms for all of the above molecules is included by necessity for calculating the atomization energies.*

#### 2 Ionization Potentials
- $\mathrm{Li} \to \mathrm{Li}^+$
- $\mathrm{C} \to \mathrm{C}^+$

#### 3 Reaction Barrier Heights (from [BH76](https://pubs.acs.org/doi/10.1021/jp045141s))
- $\mathrm{OH} + \mathrm{N_2}\to \mathrm{H}+\mathrm{N_2O}$
- $\mathrm{OH} + \mathrm{CH_3}\to\mathrm{O}+\mathrm{CH_4}$
- $\mathrm{HF} + \mathrm{F} \to \mathrm{H} + \mathrm{F_2}$

Plotted graphs show evaluation of the models on the validation set:
#### Validation AE
- $\mathrm{S_2}$
- $\mathrm{C_2H_2}$
- $\mathrm{BeH}$
- $\mathrm{NO_2}$
- $\mathrm{CH_4}$
- $\mathrm{PF_3}$
- $\mathrm{CH_2}$
- $\mathrm{CO_2HCH_3}$
#### Validation BH
- $\mathrm{H + N_2O} \to \mathrm{HON_2}$
- $\mathrm{OH + Cl} \to \mathrm{ClHO}$

#### File Summary
- `18_40c_valloss_last.png` contains plots showing the evaluated network's predicted AE and total energies, compared to reference (literature or CCSD(T)) and XCDiff. The last generated model's evaluation here is plotted.
- `18_40c_valloss_epochs.png` contains plots showing the evaluated AE and total energies as the epochs increase.
- `18_40c_valloss_rho.png` contains a plot showing the validation dataset index vs. the density loss for each model.
- `18_40c_valloss_bh1.png` contains a plot of the deviation in predicted barrier heights for the H+N2O -> HON2 reaction, along with the deviation from XCDiff's prediction
- `18_40c_valloss_bh2.png` contains a plot of the deviation in predicted barrier heights for the OH+Cl -> HClO reaction, along with the deviation from XCDiff's prediction
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
            - **Files with `no0` in the name plot the same as above, just with no zeroing of the first value.**