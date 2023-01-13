# Summary of Scripts Folder

This directory contains the various scripts for:
- generating training data
    - `run_pyscf.py`
- preparing that data for the actual training process
    - `prep_data.py` 
- using that data to train
    - `train.py`
- plotting the enhancement factors of a given model
    - `plot_enhancement.py`
- using a model to evaluate a given trajectory
    - `eval.py`
- `example_submissions` just have typical cluster submission scripts to illustrate how I invoke the scripts

# Script Flow

The general flow is as follows:
- Some set of training data is stored in an ASE trajectory.
- `run_pyscf.py` is run on the given trajectory to generate the CCSD(T) density matrices
    - e.g. `python ~/research/run_pyscf.py ./subat.traj --xc ccsdt --serial --forcepol`
    - This generates a new trajectory `results.traj` that contain the CCSD(T) energies as keys in each `Atoms.calc.results` object in the trajectory
        - e.g. `atoms.calc.results['energy']` contains the final energy, but `.results['e_hf']`, `.results['e_ccsd']`, and `.results['e_ccsdt']` are also contained
    - Density matrices/results are prefixed as `'{}_{}.format(idx, atoms.symbols)`
- The `results.traj` trajectory and the `subat.traj` trajectory are combined into a new trajectory `subat_ref.traj`. Each molecule/atom in the trajectory needs to have specific values present in its `.info` dictionary
    - `x.info['target_energy']` needs to be specified -- this is the reference energy (in Hartrees) for the loss that will be used.
        - specifically, `target_energy` is the total atomic energy for single atoms, the atomization energy for the molecules from the G2/97 dataset, and the barrier height for the reaction pathways.
    - For molecules in a reaction, `.info['reaction']` must also be specified.
        - Reactants must have `.info['reaction'] = 'reactant'`, and products must have `.info['reaction'] = INT`, where `INT` is the number of molecules in the reaction that produce the product
            - In the training set, there's only A+B -> AB or A -> A, so for the former `INT = 2` and the latter `INT = 1`
        - **NOTE: As structured, the barrier height `target_energy` must be placed on the final atom in the pathway, and the participating molecules/atoms in the pathway must come sequentially before the final molecule.**
        - **The 'target_energy' must be on the final molecule in the pathway.**
            - The `INT` is used to backtrack through the indices in the trajectory to select the reaction participants, hence why they must be sequential
    - `calc_energy` needs to be specified -- this is just the energy coming from the CCSD(T) calculations
    - This is how I combine the trajectories:
    ```
    from ase.io import read, write
    calcs = read(calcpath, ':')
    sub = read(subatpath, ':')
    for idx, i in enumerate(sub):
        #select corresponding result from calculation with run_pyscf.py
        calc = calcs[idx]
        print('+++', idx, calc, calc.calc.results['energy'])
        #ref atomization or total atomic energy, assumed to already be in subatpath trajectory's info dictionaries
        e = i.info['target_energy']
        eH = e #e in hartree
        eV = e*Hartree #e in eV
        print("++", e, eH)
        i.info['energy'] = eH
        i.info['atomization'] = eH
        i.info['atomization_ev'] = eV
        i.info['atomization_H'] = eH
        print("---", i, e)
        #copy info dictionaries to calc traj
        calc.info = i.info
        #add calculated energy from calc.results to info
        calc.info['energy'] = calc.calc.results['energy']
        #overwrite calculator to use reference energy
        calc.calc.results['energy'] = e
        if len(calc.positions) > 1:
            #if greater than 1 atom, use total energy from calculation
            #and reference atomization for target
            calc.info['target_energy'] = e
            calc.info['calc_energy'] = calc.info['energy']
            calc.info['e_calc'] = calc.info['energy']
        else:
            #else use reference energy for single atom
            try:
                calc.info['target_energy'] = i.calc.results['energy']
                calc.info['calc_energy'] = i.calc.results['energy']
                calc.info['e_calc'] = i.calc.results['energy']
            except:
                #no calc on single atoms in reaction pathways, just use ccsdt energies
                calc.info['target_energy'] = calc.info['energy']
                calc.info['calc_energy'] = calc.info['energy']
                calc.info['e_calc'] = calc.info['energy']

        calc.info['atomization'] = eH
        calc.info['atomization_ev'] = eV
        calc.info['atomization_H'] = eH
        print("--", calc.calc.results, calc.info)
    write(rp, calcs)
    ```
- `prep_dat.py` is run on this combined trajectory to generate the matrices used during training
    - `python ~/dpyscfl/scripts/prep_data.py . SCAN ./subat_ref.traj --forcepol --mingridlevel 5 --ref_path /home/awills/Documents/Research/swxcd/aegis/test_subset_ps2/07`
    - `prep_dat.py` takes the `'target_energy'` and `'calc_energy'` keys and stores them in the matrices as `'e_base'` and `'e_calc'` respectively
    The reference density matrix is read is from the reference directory and stored as `dm` in the matrices
- `train.py` can now be used to train on the data generated in this directory
    ```
    python /gpfs/home/awills/dpyscfl/scripts/train.py \
    --pretrain_loc /gpfs/home/awills/dpyscfl/models/pretrained/scan \
    --type MGGA --datapath /gpfs/scratch/awills/swxcd/data/test_subset_ps2/$DIR \
    --reftraj /gpfs/scratch/awills/swxcd/aegis/test_subset_ps2/$DIR/subat_ref.traj \
    --targetdir /gpfs/scratch/awills/swxcd/eval/test_subset_ps2/$DIR \
    --noxcdiffpop --logpath . --passthrough --chkptmax 20
    ```
- `eval.py` can be used with the resulting models to make predictions on new trajectories
    ```
    for dir in MODEL_MGGA_e*/; do cd $dir; mkdir val; cd val;
    python ~/dpyscfl/scripts/eval.py --type MGGA --xc SCAN --refpath /gpfs/scratch/awills/swxcd/aegis/validation \
    --reftraj val_c.traj --modelpath /gpfs/scratch/awills/swxcd/eval/test_subset_ps2/17/$dir/ \
    --writeeach preds --writepred --maxcycle 250 --atomization --gridlevel 5 --forceUKS \
    --gridlevel 5 2>&1 | tee eval.out;
    ```