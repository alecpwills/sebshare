This directory contains the evaluation/training outputs from training the *pre-trained SCAN model* with *default `train.py` inputs* on the test training subset:
- H2
- N2
- LiF
- F
- H
- Li
- N

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

- `01_bhval.png` contains the predicted validation barrier heights using both xcdiff and the last model generated in training
- `01_trainloss.png` contains a plot of the training losses at each epoch. Notably, the total loss does indeed improve starting with epoch `e02` and onwards.
- `01_valloss_epoch09.png` contains plots showing the evaluated network's predicted AE and total energies, compared to reference (literature or CCSD(T)) and XCDiff. The last generated model's evaluation here is plotted.
- `01_valloss_epochs.png` contains plots showing the evaluated AE and total energies as the epochs increase. 
- `MODEL_MGGA_e*/` are the evaluation directories -- eXX is the XXth model generated during training.
    - `MODEL_MGGA_e00` is just a copy of the pre-trained SCAN network before evaluation. `MODEL_MGGA_e02` is the first updated network that had lower total training loss.
- `train` has the training outputs from the training process.