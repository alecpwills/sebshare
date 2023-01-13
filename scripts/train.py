#!/usr/bin/env python
# coding: utf-8
#meant for use on local machine for ease of changes without git pulls/commits
from unittest import skip
import torch
torch.autograd.set_detect_anomaly(True)
import numpy as np
from ase.io import read
from dpyscfl.net import *
from dpyscfl.scf import *
from dpyscfl.utils import *
from dpyscfl.losses import *
from functools import partial
from ase.units import Bohr
from datetime import datetime
import pylibnxc
import os, psutil, tarfile, argparse, json, shutil
import tqdm, inspect
#from eval import KS, eval_xc, eval_wrap
old_print = print
def new_print(*args, **kwargs):
    # if tqdm.tqdm.write raises error, use builtin print
    try:
        tqdm.tqdm.write(*args, **kwargs)
    except:
        old_print(*args, ** kwargs)
# globaly replace print with new_print
inspect.builtins.print = new_print
#Validation Imports
#from eval import KS, eval_xc

process = psutil.Process(os.getpid())
#dpyscf_dir = os.environ.get('DPYSCF_DIR','..')
DEVICE = 'cpu'

parser = argparse.ArgumentParser(description='Train xc functional')
parser.add_argument('--pretrain_loc', action='store', type=str, help='Location of pretrained models (should be directory containing x and c)')
parser.add_argument('--type', action='store', choices=['GGA','MGGA'])
parser.add_argument('--datapath', action='store', type=str, help='Location of precomputed matrices (run prep_data first)')
parser.add_argument('--reftraj', action='store', type=str, help='Location of reference trajectories')
parser.add_argument('--n_hidden', metavar='n_hidden', type=int, default=16, help='Number of hidden nodes (16)')
parser.add_argument('--hyb_par', metavar='hyb_par', type=float, default=0.0, help='Hybrid mixing parameter (0.0)')
parser.add_argument('--E_weight', metavar='e_weight', type=float, default=0.01, help='Weight of total energy term in loss function (0)')
parser.add_argument('--rho_weight', metavar='rho_weight', type=float, default=20, help='Weight of density term in loss function (25)')
parser.add_argument('--ae_weight', metavar='ae_weight', type=float, default=1, help='Weight of AE term in loss function (1)')
parser.add_argument('--modelpath', metavar='modelpath', type=str, default='', help='Net Checkpoint location to continue training')
parser.add_argument('--optimpath', metavar='optimpath', type=str, default='', help='Optimizer Checkpoint location to continue training')
parser.add_argument('--logpath', metavar='logpath', action='store', type=str, default='log/', help='Logging directory (log/)')
parser.add_argument('--testrun', action='store_true', help='Do a test run over all molecules before training')
parser.add_argument('--lr', metavar='lr', type=float, action='store', default=0.0001, help='Learning rate (0.0001)')
parser.add_argument('--l2', metavar='l2', type=float, default=1e-6, help='Weight decay (1e-8)')
parser.add_argument('--hnorm', action='store_true', help='Use H energy and density in loss')
parser.add_argument('--print_stdout', action='store_true', help='Print to stdout instead of logfile')
parser.add_argument('--print_names', action='store_true', help='Print molecule names during training')
parser.add_argument('--nonsc_weight', metavar='nonsc_weight',type=float, default=0.01, help='Loss multiplier for non-selfconsistent datapoints')
parser.add_argument('--start_converged', action='store_true', help='Start from converged density matrix')
parser.add_argument('--scf_steps', metavar='scf_steps', type=int, default=25, help='Number of scf steps')
parser.add_argument('--polynomial', action='store_true', help='Use polynomials instead of neural networks')
parser.add_argument('--free', action='store_true', help='No LOB and UEG limit')
parser.add_argument('--freec', action='store_true', help='No LOB and UEG limit for correlation')
parser.add_argument('--meta_x', metavar='meta_x',type=float, default=None, help='')
parser.add_argument('--rho_alt', action='store_true', help='Alternative rho loss on total density')
parser.add_argument('--radical_factor', metavar='radical_factor',type=float, default=1.0, help='')
parser.add_argument('--forcedensloss', action='store_true', default=False, help='Make training use density loss.')
parser.add_argument('--forceEloss', action='store_true', default=False, help='Make training use TOTAL energy loss. Ill-advised to use given atomization energy structure.')
parser.add_argument('--freezeappend',  type=int, action='store', default=0, help='If flagged, freezes network and adds N duplicate layers between output layer and last hidden layer. The new layer is not frozen.')
parser.add_argument('--loadfa', type=int, action='store', default=0, help='If loading model that has appended layers, specify number of inserts between previous final and output layers here.')
parser.add_argument('--outputlayergrad', action='store_true', default=False, help='Only works with freezeappend. If flagged, sets the output layer to also be differentiable.')
parser.add_argument('--checkgrad', action='store_true', default=False, help='If flagged, executes loop over scf.xc parameters to print gradients')
parser.add_argument('--testmol', type=str, action='store', default='', help='If specified, give symbols/formula/test label for debugging purpose')
parser.add_argument('--torchtype', type=str, default='float', help='float or double')
parser.add_argument('--testall', action='store_true', default=False, help='If flagged, forces testing of entire training set.')
parser.add_argument('--targetdir', action='store', type=str, default='', help='Directory in which to store checkpoint files during training.')
parser.add_argument('--valtraj', type=str, action='store', default='', help="Path to validation trajectory. Validation won't occur without specifying this.")
parser.add_argument('--valbasis', metavar='basis', type=str, nargs = '?', default='6-311++G(3df,2pd)', help='validation basis to use. default 6-311++G(3df,2pd)')
parser.add_argument('--valpol', action='store_true', default=True, help='If flagged, force pyscf method to be UKS for validation.')
parser.add_argument('--valgridlevel', action='store', type=int, default=5, help='grid level')
parser.add_argument('--valmaxcycle', action='store', type=int, default=100, help='max cycle for validation')
parser.add_argument('--noxcdiffpop', action='store_false', default=True, help='If flagged, does NOT pop the molecules that Sebastian popped from his training set.')
parser.add_argument('--testpop', action='store_true', default=False, help='for testing purposes')
parser.add_argument('--passthrough', action='store_true', default=False, help='If flagged, first passthrough of the training trajectory just generates losses and does not update the network until the next pass.')
parser.add_argument('--subset', action='store_true', default=False, help='If flagged, will use a subset of the dataloader to loop over. Reduces overhead of reading useless files.')
parser.add_argument('--chkptmax', action='store', default=999999999, type=int, help='If specified, will not continue training after this many checkpoints.')
args = parser.parse_args()

ttypes = {'float' : torch.float,
            'double': torch.double}

ueg_limit = not args.free
HYBRID = (args.hyb_par > 0.0)


torch.set_default_dtype(ttypes[args.torchtype])

#spins for single atoms, since pyscf doesn't guess this correctly.
spins_dict = {
    'Al': 1,
    'B' : 1,
    'Li': 1,
    'Na': 1,
    'Si': 2 ,
    'Be':0,
    'C': 2,
    'Cl': 1,
    'F': 1,
    'H': 1,
    'N': 3,
    'O': 2,
    'P': 3,
    'S': 2
}

def get_spin(at):
    #if single atom and spin is not specified in at.info dictionary, use spins_dict
    if ( (len(at.positions) == 1) and not ('spin' in at.info) ):
        spin = spins_dict[str(at.symbols)]
    else:
        if at.info.get('spin', None):
            print('Spin specified in atom info.')
            spin = at.info['spin']
        elif 'radical' in at.info.get('name', ''):
            print('Radical specified in atom.info["name"], assuming spin 1.')
            spin = 1
        elif at.info.get('openshell', None):
            print("Openshell specified in atom info, attempting spin 2.")
            spin = 2
        else:
            print("No specifications in atom info to help, assuming no spin.")
            spin = 0
    return spin

def KS(mol, method, model_path='', nxc_kind='grid', **kwargs):
    """ Wrapper for the pyscf RKS and UKS class
    that uses a libnxc functionals
    """
    #hyb = kwargs.get('hyb', 0)
    mf = method(mol, **kwargs)
    if model_path != '':
        if nxc_kind.lower() == 'atomic':
            model = get_nxc_adapter('pyscf', model_path)
            mf.get_veff = veff_mod_atomic(mf, model)
        elif nxc_kind.lower() == 'grid':
            parsed_xc = pylibnxc.pyscf.utils.parse_xc_code(model_path)
            dft.libxc.define_xc_(mf._numint,
                                 eval_xc,
                                 pylibnxc.pyscf.utils.find_max_level(parsed_xc),
                                 hyb=parsed_xc[0][0])
            mf.xc = model_path
        else:
            raise ValueError(
                "{} not a valid nxc_kind. Valid options are 'atomic' or 'grid'"
                .format(nxc_kind))
    return mf

def eval_xc(xc_code, rho, spin=0, relativity=0, deriv=1, verbose=None):
    """ Evaluation for grid-based models (not atomic)
        See pyscf documentation of eval_xc
    """
    inp = {}
    if spin == 0:
        if rho.ndim == 1:
            rho = rho.reshape(1, -1)
        inp['rho'] = rho[0]
        if len(rho) > 1:
            dx, dy, dz = rho[1:4]
            gamma = (dx**2 + dy**2 + dz**2)
            inp['sigma'] = gamma
        if len(rho) > 4:
            inp['lapl'] = rho[4]
            inp['tau'] = rho[5]
    else:
        rho_a, rho_b = rho
        if rho_a.ndim == 1:
            rho_a = rho_a.reshape(1, -1)
            rho_b = rho_b.reshape(1, -1)
        inp['rho'] = np.stack([rho_a[0], rho_b[0]])
        if len(rho_a) > 1:
            dxa, dya, dza = rho_a[1:4]
            dxb, dyb, dzb = rho_b[1:4]
            gamma_a = (dxa**2 + dya**2 + dza**2)  #compute contracted gradients
            gamma_b = (dxb**2 + dyb**2 + dzb**2)
            gamma_ab = (dxb * dxa + dyb * dya + dzb * dza)
            inp['sigma'] = np.stack([gamma_a, gamma_ab, gamma_b])
        if len(rho_a) > 4:
            inp['lapl'] = np.stack([rho_a[4], rho_b[4]])
            inp['tau'] = np.stack([rho_a[5], rho_b[5]])

    parsed_xc = pylibnxc.pyscf.utils.parse_xc_code(xc_code)
    total_output = {'v' + key: 0.0 for key in inp}
    total_output['zk'] = 0
    #print(parsed_xc)
    for code, factor in parsed_xc[1]:
        model = pylibnxc.LibNXCFunctional(xc_code, kind='grid')
        output = model.compute(inp)
        for key in output:
            if output[key] is not None:
                total_output[key] += output[key] * factor

    exc, vlapl, vtau, vrho, vsigma = [total_output.get(key,None)\
      for key in ['zk','vlapl','vtau','vrho','vsigma']]

    vxc = (vrho, vsigma, vlapl, vtau)
    fxc = None  # 2nd order functional derivative
    kxc = None  # 3rd order functional derivative
    return exc, vxc, fxc, kxc


def get_singles(atoms, model, valbasis, valpol, valgridlevel, maxcycle):
    atomic_set = []
    for at in atoms:
        atomic_set += at.get_chemical_symbols()
    atomic_set = list(set(atomic_set))
    for s in atomic_set:
        assert s in list(spins_dict.keys()), "{}: Atom in dataset not present in spins dictionary.".format(s)
    atomic_e = {s:0 for s in atomic_set}
    atomic = [Atoms(symbols=s) for s in atomic_set]
    #generates pyscf mol, default basis 6-311++G(3df,2pd), charge=0, spin=None
    atomic_mol = [ase_atoms_to_mol(at, basis=valbasis, spin=get_spin(at), charge=0) for at in atomic]
    atomic_method = [gen_mf_mol(mol[1], xc='notnull', pol=valpol, grid_level=valgridlevel, nxc=True) for mol in atomic_mol]
    for idx, methodtup in enumerate(atomic_method):
        print(idx, methodtup)
        name, mol = atomic_mol[idx]
        method = methodtup[1]
        mf = KS(mol, method, model_path=model)
            
        mf.grids.level = valgridlevel
        mf.max_cycle = maxcycle
        #mf.density_fit()
        mf.kernel()

        atomic_e[name] = mf.e_tot
        print("++++++++++++++++++++++++")
        print("Atomic Energy: {} -- {}".format(name, mf.e_tot))
        print("++++++++++++++++++++++++")
    with open('atomicen.val.dat', 'w') as f:
        f.write("#Atom \t Energy (Hartree) \n")
        for k, v in atomic_e.items():
            f.write('{} \t {} \n'.format(k, v))
            print('{} \t {}'.format(k,v))
    with open('atomicen.val.pkl', 'wb') as f:
        pickle.dump(atomic_e, f)
    return atomic_e

def get_validation(atoms, model, valbasis, valpol, valgridlevel, maxcycle):
    energies = {at.get_chemical_formula():0 for at in atoms}

    for idx, atom in enumerate(atoms):
        formula = atom.get_chemical_formula()
        symbols = atom.symbols
        results = {}
        #manually skip for preservation of reference file lookups

        print("================= {}:    {} ======================".format(idx, formula))
        print("Getting Validation Datapoint")
        molgen = False
        scount = 0
        while not molgen:
            try:
                name, mol = ase_atoms_to_mol(atom, basis=valbasis, spin=get_spin(atom)-scount, charge=0)
                molgen=True
            except RuntimeError:
                #spin disparity somehow, try with one less until 0
                print("RuntimeError. Trying with reduced spin.")
                spin = get_spin(atom)
                spin = spin - scount - 1
                scount += 1
                if spin < 0:
                    raise ValueError
        _, method = gen_mf_mol(mol, xc='notnull', pol=valpol, grid_level=valgridlevel, nxc=True)
        mf = KS(mol, method, model_path=model)
        mf.grids.level = valgridlevel
        mf.grids.build()
        #mf.density_fit()
        mf.max_cycle = maxcycle
        mf.kernel()

        energies[formula] = mf.e_tot

    return energies



def scf_wrap(scf, dm_in, matrices, sc, molecule=''):
    try:
        results = scf(dm_in, matrices, sc)
    except Exception as e:
        print("========================================================")
        print("EEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE")
        print("SCF CALCULATION FAILED")
        print("SCF Calculation failed for {}".format(molecule))
        print("{}".format(e))
        print("EEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE")
        print("========================================================")
        results = None
    return results


def get_optimizer(model, path='', hybrid=HYBRID):
    if hybrid:
            optimizer = torch.optim.Adam(list(model.parameters()) + [model.xc.exx_a],
                                    lr=args.lr, weight_decay=args.l2)
    else:
        optimizer = torch.optim.Adam(model.parameters(),
                                    lr=args.lr, weight_decay=args.l2)

    MIN_RATE = 1e-7
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',
                                                            verbose=True, patience=int(10/PRINT_EVERY),
                                                            factor=0.1, min_lr=MIN_RATE)

    if path:
        optimizer.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
    return optimizer, scheduler


if __name__ == '__main__':

    timestamp = str(datetime.now()).replace(' ','_')
    logpath = os.path.join(args.logpath, timestamp)

    if not args.print_stdout:
        def print(*args):
            with open(logpath + '.log','a') as logfile:
                logfile.write(' ,'.join([str(a) for a in args]) + '\n')

    try:
        os.mkdir('/'.join(logpath.split('/')[:-1]))
    except FileExistsError:
        pass

    print(json.dumps(args.__dict__,indent=4))
    with open(logpath+'.config','w') as file:
        file.write(json.dumps(args.__dict__,indent=4))

    with tarfile.open(logpath + '.tar.gz', "w:gz") as tar:
        #source_dir = dpyscf_dir + '/dpyscf/'
        #tar.add(source_dir, arcname=os.path.basename(source_dir))
        source_dir = __file__
        tar.add(source_dir, arcname=os.path.basename(source_dir))

    print("READING REFERENCE TRAJECTORY.")
    atoms = read(args.reftraj, ':')
    indices = np.arange(len(atoms)).tolist()
    #Some molecules in the trajectory are not needed to reproduce xcdiff training set.
    #But don't pop them, because indexing important.
    #skips = ['O2', 'Cl2', 'HCl']
    #skips = ['O2']
    skips = []
    #pop = []
    
    #xcdiff pops
    if args.noxcdiffpop:
        pop = [34, 33, 32, 10, 7, 5]
    else:
        pop = []
    if args.testpop:
        pop = list(np.arange(len(atoms)-1))
    #print("popping specified atoms: {}".format(pop))
    #[atoms.pop(i) for i in pop]
    #[indices.pop(i) for i in pop]


    print("READING DATASET")

    #dataset = MemDatasetRead(args.datapath, skip=pop)
    dataset = MemDatasetRead(args.datapath, skip=[])
    dataset_train = dataset

    print("LOADING DATASET INTO PYTORCH")
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False) # Dont change batch size !
    dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=1, shuffle=False) # Dont change batch size !
    
    print("PARSING NON-ATOMIC NON-REACTION MOLECULES")
    molecules = {'{:3d}'.format(idx) + ''.join(a.get_chemical_symbols()): [idx] for idx, a in enumerate(atoms) if len(a.positions) > 1 and not a.info.get('reaction') }
    print(molecules)

    print("PARSING NEUTRAL, PURE NON-REACTION ATOMS. CHARGE FLAG NOT SET.")
    pure_atoms = {''.join(a.get_chemical_symbols()): [idx] for idx, a in enumerate(atoms) if len(a.positions) == 1 and not a.info.get('reaction') and not a.info.get('fractional') and not a.info.get('charge') and not a.info.get('supp')}
    print(pure_atoms)
    molecules.update(pure_atoms)
    
    print("PARSING SUPPLEMENTAL NEUTRAL, PURE ATOMS (FROM FRAC DATASET)")
    n_atoms = {''.join(a.get_chemical_symbols())+'_n0': [idx] for idx, a in enumerate(atoms) if len(a.positions)==1 and a.info.get('supp') and not a.info.get('charge') and not a.info.get('fractional')}
    print(n_atoms)

    #molecules.update(n_atoms)
    print("PARSING SUPPLEMENTAL CHARGED, PURE ATOMS")
    c_atoms = {''.join(a.get_chemical_symbols())+'_c{}'.format(a.info['charge']): [idx] for idx, a in enumerate(atoms) if len(a.positions)==1 and a.info.get('supp') and a.info.get('charge')}
    print(c_atoms)

    #molecules.update(c_atoms)
    print("PARSING SUPPLEMENTAL FRACTIONAL ATOMS")
    frac_atoms = {''.join(a.get_chemical_symbols())+'_f{}'.format(a.info['fractional']): [idx] for idx, a in enumerate(atoms) if len(a.positions)==1 and a.info.get('supp') and a.info.get('fractional')}
    print(frac_atoms)
    #molecules.update(frac_atoms)
    
    def cat_dict(dicts, keysplit='_'):
        #To generate the list of atoms comprising a fractional datapoint, for effective reaction
        retdct = {k:[] for k in list(dicts[0].keys())}
        rkeys = sorted(list(retdct.keys()))
        for didx,dct in enumerate(dicts):
            dkeys = sorted(list(dct.keys()))
            for dk in dkeys:
                mkey = [mk for mk in rkeys if dk.split(keysplit)[0] in mk.split(keysplit)[0]][0]
                retdct[mkey] += dct[dk]
        return retdct

    fracdct = cat_dict([frac_atoms, n_atoms, c_atoms])
    print("CONCATENATING SUPPLEMENTAL/FRACTIONAL ATOMS")
    print(frac_atoms)
    molecules.update(fracdct)

    def split(el):
            import re
            #Splits a string on capital letter sequences
            res_list = [s for s in re.split("([A-Z][^A-Z]*)", el) if s]
            return res_list

    for molecule in molecules:
        comp = []
        #ignore _ atoms, charged or fractional
        if '_' in molecule:
            continue
        for a in split(molecule[3:]):
            comp.append(pure_atoms[a][0])
        molecules[molecule] += comp

    
    print("PARSING REACTIONS")
    reactions = {}
    for idx, a in enumerate(atoms):
        #atom must have reaction flag
        if not a.info.get('reaction'): continue
        #and atom must not be a reactant
        if a.info.get('reaction') == 'reactant': continue
        reactions['{:3d}'.format(idx) + ''.join(a.get_chemical_symbols())] = \
            [idx] + [idx + i for i in np.arange(-a.info.get('reaction'),0,1).astype(int)]
    print(reactions)
    molecules.update(reactions)


    #a_count = {idx: len(at.positions) for idx,at in enumerate(atoms)}
    #count number of times each standalone atom/molecule appears in dataset -- most things are just once, but the single atoms appear in larger molecules so count more
    a_count = {a: np.sum([a in molecules[mol] for mol in molecules]) for a in np.unique([m  for mol in molecules for m in molecules[mol]])}

    print("MOLECULES TO TRAIN ON")
    print(molecules)
    print("A_COUNT")
    print(a_count)

    best_loss = 1e6
    print("GENERATING SCF OBJECT")
    if args.pretrain_loc:
        scf = get_scf(args.type, pretrain_loc=args.pretrain_loc, hyb_par=args.hyb_par)
    elif args.modelpath:
        scf = get_scf(args.type, path=args.modelpath, inserts=args.loadfa, hyb_par=args.hyb_par)
    else:
        scf = get_scf(args.type, hyb_par=args.hyb_par)
    scf.nsteps = args.scf_steps

    if args.freezeappend:
        print("\n ================================= \n")
        print("FREEZING SCF MODEL.XC AND APPENDING NEW LAYER")
        print("\n ================================= \n")
        freeze_append_xc(scf, args.freezeappend, args.outputlayergrad)

    if args.testrun:
        print("\n ======= Starting testrun ====== \n\n")
        with open(logpath+'testrun.dat', 'w') as f:
            f.write('#IDX\t FORMULA\t SYMBOLS\t E_PRETRAINED_MODEL\t CONVERGENCE\t SC\n')

        #Set SCF Object training flag off
        scf.xc.evaluate()

        Es = []
        E_pretrained = []
        tested = []
        #TODO: fix how things are loaded to use testrun
        #for dm_init, matrices, e_ref, dm_ref in dataloader_train:
        #for dm_init, matrices in dataloader_train:
        #TODO: way data is loaded makes this incorrect -- e_ref is the atomization, not total energy
        for tridx, data in enumerate(dataloader_train):
            atom = atoms[tridx]
            cf, cs = (atom.get_chemical_formula(), str(atom.symbols))
            sc = atom.info.get('sc',True)

            #If atom not self-consistent, skip
            if args.testall:
                print("Testing all molecules/atoms in training data.")
            elif not sc:
                print('skipping {}, not sc'.format(cf))
                continue
            elif ( (cf in skips) or (cs in skips) ):
                print("skipping {}, in skips".format(atom.get_chemical_formula()))
                continue
            tested.append(atom)
            print("====================================")
            print("Testrun Calculation")
            print(tridx, atom, cf, cs)
            print("====================================")
            dm_init = data[0]
            matrices = data[1]
            
            try:
                #previous prep_data had different keys for the matrix values
                e_ref = matrices['e_base']
            except KeyError:
                print("Wrong key, trying Etot from matrices")
                e_ref = matrices['Etot']

            dm_ref = matrices['dm']
            dm_init = dm_init.to(DEVICE)
            e_ref = e_ref.to(DEVICE)
            dm_ref = dm_ref.to(DEVICE)
            matrices = {key:matrices[key].to(DEVICE) for key in matrices}

            results = scf_wrap(scf, matrices['dm_realinit'], matrices, sc, molecule='testrun_{}'.format(tridx))
            if not results:
                continue

            #E_pretrained.append(matrices['e_base'])
            E = results['E']

            if sc:
                tmpE = E.detach().cpu().numpy()
                Es.append(E.detach().cpu().numpy())
            else:
                tmpE = [E.detach().cpu().numpy()[0]]*scf.nsteps
                Es.append(np.array([E.detach().cpu().numpy()]*scf.nsteps))
            tmpC = tmpE[-1] - tmpE[-2]

            with open(logpath+'testrun.dat', 'a') as f:
                f.write('{}\t {}\t {}\t {}\t {}\t {}\n'.format(tridx, cf, cs, tmpE[-1], tmpC, sc))

        # e_premodel = np.array(Es)[:,-1]
        # #error_pretrain = e_premodel - np.array(E_pretrained)
        # convergence = np.array(Es)[:,-1]-np.array(Es)[:,-2]
        # print("\n ------- Statistics ----- ")
        # print(str(e_premodel), 'Energies from model' )
        # #print(str(np.array(E_pretrained)),'Energies from exact DFT baseline')
        # #print(str(error_pretrain), 'Pretraining error')
        # print(str(convergence), 'Convergence')

        # with open(logpath+'testrun2.dat', 'a') as f:
        #     f.write('#IDX FORMULA SYMBOLS E_PRETRAINED_MODEL CONVERGENCE SC\n')
        #     for i in range(len(tested)):
        #         atom = tested[i]
        #         sc = atom.info.get('sc', True)
        #         cf, cs = (atom.get_chemical_formula(), str(atom.symbols))
        #         if args.testall:
        #             print("Testall flagged")
        #         elif not sc:
        #             print('non-sc atom {}, skipping')
        #         elif ( (cf in skips) or (cs in skips) ):
        #             print("write test: skipping {}".format(atom.get_chemical_formula()))
        #             continue
        #         f.write('{}\t {}\t {}\t {}\t {}\t {}\n'.format(i, cf, cs, e_premodel[i], convergence[i], sc))

            


    print("\n ======= Starting training ====== \n\n")
    scf.xc.train()
    PRINT_EVERY=1
    skip_steps = max(5, args.scf_steps - 10)

    optimizer, scheduler = get_optimizer(scf, path=args.optimpath)

    #Loss Functions -- Density
    density_loss = rho_alt_loss if args.rho_alt else rho_loss
    # args.rho_weight defaults to 20, per the paper
    # args.E_weight defaults to 0.01, per the paper
    # AE Loss has weight 1 by default, per the paper, but decreases for non-sc
    mol_losses = {"rho" : (partial(density_loss, loss = torch.nn.MSELoss()), args.rho_weight)}
    atm_losses = {"E":  (partial(energy_loss, loss = torch.nn.MSELoss()), args.E_weight)}
    h_losses = {"rho" : (partial(density_loss,loss = torch.nn.MSELoss()), args.rho_weight),
                "E":  (partial(energy_loss, loss = torch.nn.MSELoss()), args.E_weight)}

    ae_loss = partial(ae_loss,loss = torch.nn.MSELoss())
     
    #Indices for self-consistent training molecules
    train_order = np.arange(len(molecules)).astype(int)
    molecules_sc = {}
    for m_idx in train_order:
        molecule = list(molecules.keys())[m_idx]
        mol_sc = True
        for idx in range(len(dataloader_train)):
            if not idx in molecules[molecule]: continue
            mol_sc = atoms[idx].info.get('sc',True)
        molecules_sc[molecule] = mol_sc
        
            
    chkpt_idx = 0
    validate_every = 10

    for epoch in range(100000):
        encountered_nan = True
        if (epoch == 0 and args.passthrough):
            #will be turned to train after first epoch
            #first run through will just evaluate on the molecules and print losses
            print("FIRST PASS: EVALUATION ON TRAINING DATA")
            scf.xc.evaluate()
        elif (epoch == 1 and args.passthrough):
            print("NEXT EPOCH BEGINNING - PASSTHROUGH COMPLETE.")
            print("SETTING NETWORK TO TRAINING MODE.")
            scf.xc.train()
        if epoch == 0:
            with open(logpath+'_evalloss.dat', 'a') as f:
                f.write('#epoch\tm_idx\tidx\tat_form\tatsym\t[losskey\tloss\tloss*weight]\n')
        while(encountered_nan):
            error_cnt = 0
            running_losses = {"rho": 0, "ae":0, "E":0}
            total_loss = 0
            atm_cnt = {}
            encountered_nan = False
            try:
                #Randomly choose training order
                train_order = np.arange(len(molecules)).astype(int)
                np.random.shuffle(train_order)
                t = tqdm.tqdm(train_order)
                fails = []
                #loop over tqdm object for shell progress bar
                for m_idx in t:
#                for m_idx in train_order:
                    molecule = list(molecules.keys())[m_idx]
                    submolecules = [atoms[idx].get_chemical_formula() for idx in molecules[molecule]]
                    print("================================")
                    print('--------------{}----------------'.format(m_idx))
                    print("TRAINING ON MOLECULE: ", molecule)
                    print("SUBMOLECULES: {}".format(submolecules))
                    print("SUBMOLECULE INDICES: {}".format(molecules[molecule]))
                    print("================================")
                    #if molecule not self-consistent and the weight associated to nonsc molecules is 0, skip it
                    if not molecules_sc[molecule] and not args.nonsc_weight: continue
                    
                    m_form = atoms[molecules[molecule][0]].get_chemical_formula()
                    m_sym = atoms[molecules[molecule][0]].symbols
                    if args.testmol:
                        if not ( (args.testmol == m_form) or (args.testmol in molecule) ):
                            continue
                    if (m_form in skips) or (m_sym in skips) or (m_idx in pop):
                        print("SKIPPING: ", molecule)
                        continue
                    mol_sc = True
                    ref_dict = {}
                    pred_dict = {}
                    loss = 0
                    #subset Dataset so that we don't have to load unnecessary data during one molecule step
                    #previously, this looped over the entire Dataset and matched indices contained in molecule list
                    
                    #TEST: loop over everything, not subset
                    if args.subset:
                        print("Subsetting Dataset with molecules[{}]: ".format(molecule), molecules[molecule])
                        subset = torch.utils.data.Subset(dataset, molecules[molecule])
                        subset_loader = torch.utils.data.DataLoader(subset, batch_size=1, shuffle=False)
                        loader = subset_loader
                    else:
                        loader = dataloader_train
                    for idx, data in enumerate(loader):
                    #for didx, data in enumerate(subset_loader):
                        if args.subset:
                            print("Subset: Reassigning didx <- idx; idx <- molecules[molecule][didx]")
                            didx = idx
                            idx = molecules[molecule][didx]
                            print(didx, idx)

                        #skip non-relevant atoms for current calculation
                        if idx not in molecules[molecule]:
                            print("Irrelevant index {}. Skipping to next.".format(idx))
                            continue
                        if idx in pop:
                            print("Skipping index {} -- in pop, irrelevant molecule.".format(idx))
                            continue

                        if idx == molecule[0]:
                            print(idx, data[1].keys())
                        #modify loading bar for descriptive progress during training
                        t.set_postfix({"Epoch": epoch, 'Training Label':molecule, 'Molecules': [atoms[idx].get_chemical_formula() for idx in molecules[molecule]],
                        'Current Sub-Molecule': atoms[idx].get_chemical_formula()})

                        print("Calculating sub-atoms in molecule -- ", atoms[idx])
                        if args.print_names: print(atoms[idx])
                        #xcdiff version saved as dm_init, matrices, e_ref, dm_ref,
                        #but everything after dm_init contained in matrices now
                        dm_init = data[0]
                        matrices = data[1]

                        #LOAD REFERENCE ENERGIES
                        #For atoms in G2/97 set, this will be the atomization energy
                        #For pure atoms, this will be the total energy.
                        #For atoms in the BH76 dataset/other reactions, this is the reaction height for the end atom
                        #----> If atom is in a reaction and not the final result, this will be zero.
                        try:
                            #previous prep_data had different keys for the matrix values
                            e_ref = matrices['e_base']
                        except KeyError:
                            print("Wrong key, trying Etot from matrices")
                            e_ref = matrices['Etot']


                        #get ref dm, send extracted values to device
                        dm_ref = matrices['dm']
                        dm_init = dm_init.to(DEVICE)
                        e_ref = e_ref.to(DEVICE)
                        dm_ref = dm_ref.to(DEVICE)
                        matrices = {key:matrices[key].to(DEVICE) for key in matrices}
                        dm_mix = matrices['dm_realinit']
                        print("REFERENCE ENERGY: {}".format(e_ref))
                        
                        #if converged, don't mix dm's
                        if args.start_converged:
                            mixing = torch.rand(1)*0
                        else:
                            mixing = torch.rand(1)/2 + 0.5
                        
                        sc = atoms[idx].info.get('sc',True)

                        #mix dms if not converged/if sc
                        if sc:
                            dm_in = dm_init*(1-mixing) + dm_mix*mixing
                        else:
                            dm_in = dm_init
                            mol_sc = False

                        #get flags to determine loss -- reaction, fractional, supplemental, charge
                        reaction = atoms[idx].info.get('reaction',False)
                        fractionFlag = atoms[idx].info.get('fractional', False)
                        suppFlag = atoms[idx].info.get('supp', False)
                        charge = atoms[idx].info.get('charge', 0)
                        if suppFlag:
                            print("***************************")
                            print("SUPPLEMENTAL FLAG SET.")
                            print("INDEXED ATOM: {}, {}".format(idx, atoms[idx].get_chemical_formula()))
                            print("***************************")
                            
                            if fractionFlag:
                                print("***************************")
                                print("FRACTIONAL FLAG SET. f = {}".format(fractionFlag))
                                print("INDEXED ATOM: {}, {}".format(idx, atoms[idx].get_chemical_formula()))
                                print("***************************")
                                ##TODO: implement this in the trajectory, as opposed to post-processing
                            if suppFlag and not fractionFlag:
                                #A + B flags
                                #atom flagged as supplemental to dataset, but is not the fractionally mixed atom
                                reaction = 'reactant'
                            elif suppFlag and fractionFlag:
                                # --> AB
                                #fractionally mixed, end result of A+B->AB, so 2
                                reaction = 2

                        #CALCULATION
                        print("SCF CALCULATION")
                        results = scf_wrap(scf, dm_in, matrices, sc, molecule=molecule)
                        if results == None:
                            #failed calculation, break out of this molecule's loop and continue with next
                            fails.append({"Epoch": epoch, 'Training Label':molecule, 'Molecules': [atoms[idx].get_chemical_formula() for idx in molecules[molecule]],
                        'Current Sub-Molecule': atoms[idx].get_chemical_formula()})
                            break
                        print("E_REF: {}".format(e_ref))
                        print("E_PRED: {}".format(results['E']))
                        #Add matrix keys to results dict
                        print("Adding reference matrices to results.")
                        results['dm_ref'] = dm_ref
                        results['fcenter'] = matrices.get('fcenter',None)
                        results['rho'] = matrices['rho']
                        results['ao_eval'] = matrices['ao_eval']
                        results['grid_weights'] = matrices['grid_weights']

                        #want to enforce energy accuracy -- need to include ccsdt energy in the loss
                        #results not used in atm_loss, so can modify safely here
                        e_calc = atoms[idx].info.get('e_calc', None)
                        results['e_calc'] = e_calc if e_calc else e_ref
                        results['E_ref'] = e_ref

                        results['mo_energy_ref'] = matrices['mo_energy']
                        results['n_elec'] = matrices['n_elec']
                        results['e_ip_ref'] = matrices['e_ip']
                        results['mo_occ'] = matrices['mo_occ']


                        print("RESULTS MATRICES EXTRACTED")
                        print("================================")
                        print("RESULTS MATRICES SHAPES")
                        for k,v in results.items():
                            if k == 'fcenter':
                                #chance this isn't in matrices, so None in results, so just skip
                                continue
                            elif k == "e_calc":
                                print(results[k])
                            else:
                                print("{}   ---   {}".format(k, v.shape))
                        print("================================")

                        #If radical, multiplicative factor
                        if atoms[idx].info.get('radical', False):
                            results['rho'] *= args.radical_factor
                            results['dm'] *= args.radical_factor
                        #If molecule and self-consistent, use "mol_losses" dict
                        #Only rho loss
                        if len(atoms[idx].positions) > 1 and sc:
                            print("MOL_LOSSES")
                            losses = mol_losses
                        #Else, if chosen atom is either H or Li, and args specify use both, and the H/Li not involved in reaction
                        #Rho and energy loss
                        elif str(atoms[idx].symbols) in ['H', 'Li'] and args.hnorm and not reaction:
                            print("H_LOSSES")
                            losses = h_losses
                        #Otherwise, if just an atom not in a reaction:
                        #Only energy loss -- E_ref for atoms is total atomic, not atomization
                        elif sc and not reaction:
                            print("ATM_LOSSES")
                            losses = atm_losses
                        #Else empty loss dict if reaction or not sc
                        else:
                            losses = {}
                        #if choose to force density or e loss, manually add back in
                        if args.forcedensloss:
                            if 'rho' not in losses.keys():
                                losses['rho'] = (partial(density_loss,loss = torch.nn.MSELoss()), args.rho_weight)
                        if args.forceEloss:
                            if 'E' not in losses.keys():
                                losses['E'] = (partial(energy_loss, loss = torch.nn.MSELoss()), args.E_weight)

                        #For each key in whichever loss dict chosen,
                        #Select the function (it's a tuple of itself, its weight), feed in results dict, normalize by number of atoms
                        losses_eval = {key: losses[key][0](results)/a_count[idx] for key in losses}
                        print("LOSSES_EVAL: ", losses_eval)
                        #Update running losses with new losses
                        running_losses.update({key:running_losses[key] + losses_eval[key].item() for key in losses})
                        
                        #IF Reaction type is 2, it is an A+B -> AB reaction.
                        #Store the dataset e_ref as ref, and results E as prediction
                        if reaction == 2:
                            print("REACTION TYPE: 2. A+B -> AB")
                            ref_dict['AB'] = e_ref
                            #if sc, get last skip_steps of scf cycle energies
                            #otherwise, get energy as list
                            if sc:
                                pred_dict['AB'] = results['E'][skip_steps:]
                            else:
                                pred_dict['AB'] = results['E'][-1:]

                        #ELSE if Reaction type is 1, it is an A->A reaction with some charge difference,
                        #Typically, reactant is charged so reaction == 1 is neutral
                        elif reaction == 1:
                            print("REACTION TYPE: 1. A -> A")
                            ref_dict['AA'] = e_ref
                            #if sc, get last skip_steps of scf cycle energies
                            #otherwise, get energy as list
                            if sc:
                                pred_dict['AA'] = results['E'][skip_steps:]
                            else:
                                pred_dict['AA'] = results['E'][-1:]
                        #ELSE IF it is a reactant in either of the above pathways,
                        elif reaction == 'reactant':
                            print("REACTION TYPE: REACTANT.")
                            #If self-consistent,
                            if sc:
                                label = 'A' if not 'A' in ref_dict else 'B'
                                ref_dict[label] = e_ref
                                pred_dict[label] = results['E'][skip_steps:]
                                if fractionFlag:
                                    #Efract = (1-f)*En + f*Ec
                                    if charge == 0:                                    
                                        pred_dict[label] = (1-fractionFlag)*results['E'][skip_steps:]
                                    elif charge == 1:
                                        pred_dict[label] = (fractionFlag)*results['E'][skip_steps:]
                            else:
                                label = 'A' if not 'A' in ref_dict else 'B'
                                ref_dict[label] = e_ref
                                pred_dict[label] = results['E'][-1:]
                                if fractionFlag:
                                    if charge == 0:                                    
                                        pred_dict[label] = (1-fractionFlag)*results['E'][-1:]
                                    elif charge == 1:
                                        pred_dict[label] = (fractionFlag)*results['E'][-1:]
                        
                        #If not reaction 2, 1, reactant, and molecule has more than one atom, e_ref is reference energy
                        elif len(atoms[idx].positions) > 1:
                            ref_dict[''.join(atoms[idx].get_chemical_symbols())] = e_ref
                            if sc:
                                steps = skip_steps
                            else:
                                steps = -1    
                            pred_dict[''.join(atoms[idx].get_chemical_symbols())] = results['E'][steps:]
                        #Else if not reaction 2, 1, reactant, and is single atom, ref_en if e_ref
                        else:
                            #ref_dict[''.join(atoms[idx].get_chemical_symbols())] = torch.zeros_like(e_ref)
                            ref_dict[''.join(atoms[idx].get_chemical_symbols())] = e_ref
                            pred_dict[''.join(atoms[idx].get_chemical_symbols())] = results['E'][skip_steps:]
                        #add losses*loss_weight from dictionary
                        loss += sum([losses_eval[key]*losses[key][1] for key in losses])
                        print(loss)
                        with open(logpath+'_evalloss.dat', 'a') as f:
                            atform = atoms[idx].get_chemical_formula()
                            atsym = atoms[idx].symbols
                            wstr = '{}\t{}\t{}\t{}\t{}\t'.format(epoch, m_idx, idx, atform, atsym)
                            keys = list(losses_eval.keys())
                            for k in keys:
                                wstr += '{}\t{}\t{}\t'.format(k, losses_eval[k], losses_eval[k]*losses[k][1])
                            wstr+='\n'
                            f.write(wstr)
                    if not results:
                        continue
                    
                    print("LOOP OVER SUBMOLECULES COMPLETED")
                    print("REF_DICT: ", ref_dict)
                    print("PRED_DICT: ", pred_dict)
                    ael = ae_loss(ref_dict,pred_dict)
                    running_losses['ae'] += ael.item()
                    aelstr = '{}\t {} \t {} \t {} \t {} \t {}\n'.format(epoch, m_idx, m_form, m_sym, ael.item(), args.ae_weight*ael.item())
                    print(aelstr)
                    if epoch == 0:
                        with open(logpath+'_aeloss.dat', 'a') as f:
                            f.write('#epoch\tm_idx\tm_form\tm_sym\tael\tael*lambda_ae\n')
                    with open(logpath+'_aeloss.dat', 'a') as f:
                        f.write(aelstr)
                    if mol_sc:
                        running_losses['ae'] += ael.item() * args.ae_weight
                        loss += ael * args.ae_weight
                    else:
                        loss += args.nonsc_weight * ael
                        running_losses['ae'] += args.nonsc_weight * ael.item()
                    total_loss += loss.item()
                    if (epoch == 0 and args.passthrough):
                        #if first pass and specify args.passthrough, first pass does evaluation and not train.
                        #set here to train once first pass completes
                        print("PASSTHROUGH -- {} DONE.".format(m_form))
                    else:
                        print("Backward Propagation")
                        loss.backward()
                        if args.checkgrad:
                            for p in scf.xc.parameters():
                                if p.requires_grad:
                                    print('===========\ngradient\n----------\nmax: {}\nmin: {}'.format(torch.max(p.grad), torch.min(p.grad)))
                        print("Step Optimizer")
                        optimizer.step()
                        print("Zeroing Optimizer Grad")
                        optimizer.zero_grad()
            except RuntimeError:
                encountered_nan = True
                chkpt_idx -= 1
                print('NaNs encountered, rolling back to checkpoint {}'.format(chkpt_idx%3))
                xcpath = args.modelpath if args.modelpath else args.pretrain_loc
                if chkpt_idx == -1:
                    scf = get_scf(args.type, xcpath)
                    optimizer, scheduler = get_optimizer(scf)
                else:
                    scf = get_scf(args.type, logpath + '_{}.chkpt'.format(chkpt_idx%3))
                    optimizer, scheduler = get_optimizer(scf, logpath + '_{}.adam.chkpt'.format(chkpt_idx%3))
                scf.xc.train()
                error_cnt +=1
                if error_cnt > 3:
                    print('NaNs could not be resolved by rolling back to checkpoint')
                    raise RuntimeError('NaNs could not be resolved by rolling back to checkpoint')

        if epoch%PRINT_EVERY==0:
            print("++++++++++++++++++++++++++++++++++")
            print("FAILS:")
            for i in fails:
                print(i)
            print("++++++++++++++++++++++++++++++++++")
            if epoch == 0:
                with open(logpath+'_totallosses.dat', 'w') as f:
                    wstr = "#\tEpoch\tE\tE*lambda_E\trho\trho*lambda_rho\tae\tae*lambda_ae\ttotal\n"
                    f.write(wstr)
            running_losses = {key:np.sqrt(running_losses[key]/len(molecules))*1000 for key in running_losses}
            total_loss = np.sqrt(total_loss/len(molecules))*1000
            with open(logpath+'_totallosses.dat', 'a') as f:
                wstr = "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n".format(epoch, running_losses['E'], running_losses['E']*args.E_weight, running_losses['rho'], running_losses['rho']*args.rho_weight, running_losses['ae'], running_losses['ae']*args.ae_weight, total_loss)
                f.write(wstr)
            best_loss = min(total_loss, best_loss)
            chkpt_str = 'NOUPDATE'
            pt_str = 'NOUPDATE'
            torch.save(scf.xc.state_dict(), logpath + '_current.chkpt')
            torch.save(scf, logpath + '_current.pt')
            if total_loss == best_loss:
                torch.save(scf.xc.state_dict(), logpath + '_{}.chkpt'.format(chkpt_idx%3))
                torch.save(scf, logpath + '_{}.pt'.format(chkpt_idx%3))
                torch.save(optimizer.state_dict(), logpath + '_{}.adam.chkpt'.format(chkpt_idx%3))
                chkpt_str = '_{}.chkpt'.format(chkpt_idx%3)
                pt_str = '_{}.pt'.format(chkpt_idx%3)
                chkpt_idx += 1
                if args.targetdir:
                    #targetdir specified, generate checkpoints here in MODEL_folders
                    dirstr = 'MODEL_{}'.format(args.type)
                    esuff = '_e0{}'.format(epoch) if epoch < 10 else '_e{}'.format(epoch)
                    dirstr = os.path.join(args.targetdir, dirstr+esuff)

                    savept = timestamp+pt_str
                    savef = os.path.join(dirstr, savept)
                    savexc = os.path.join(dirstr, 'xc')
                    #make dir if not exist, otherwise assume exists and don't do anything
                    try:
                        print("os.mkdir({})".format(dirstr))
                        os.mkdir(dirstr)
                    except:
                        e = sys.exc_info()[0]
                        print(e)
                        #if directory exists, move on
                        pass
                    try:
                        print("torch saving pt:")
                        torch.save(scf, savef)
                        print('os.symlink: {} -> {}'.format(savef, savexc))
                        os.symlink(savef, savexc)
                    except:
                        e = sys.exc_info()[0]
                        print(e)
                        #either directory exists or file does not yet exist
                        pass

            print("============================================================")
            print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
            print('Epoch {} ||'.format(epoch), [' {} : {:.6f}'.format(key,val) for key, val in running_losses.items()],
                  '|| total loss {:.6f} || CHKPT: {}'.format(total_loss, pt_str))
            if HYBRID:
                print("HYB MIXING:")
                print(scf.xc.exx_a)
            print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
            print("============================================================")
            scheduler.step(total_loss)

            if args.valtraj:
                #validation -- symlink current.pt to xc
                os.symlink(logpath+'_current.pt', os.path.join(args.logpath, 'xc'))
                val = read(args.valtraj, ':')

                atms = [i for i in val if i.info.get('atomization', None)]
                bhs = [i for i in val if i.info.get('product', None)]
                products = list(set([i.info['product'] for i in bhs]))

                atomic_e = get_singles(atms, model=args.logpath, valbasis=args.valbasis,
                                        valpol = args.valpol, valgridlevel=args.valgridlevel, maxcycle=args.valmaxcycle)

                val_e = get_validation(val, model=args.logpath, valbasis=args.valbasis,
                                        valpol = args.valpol, valgridlevel=args.valgridlevel, maxcycle=args.valmaxcycle)

                for at in atms:
                    formula = at.get_chemical_formula()
                    pred_e = val_e[at.get_chemical_formula()]
                    start = pred_e
                    subs = at.get_chemical_symbols()
                    if len(subs) == 1:
                        #single atom, no atomization needed
                        print("SINGLE ATOM -- NO ATOMIZATION CALCULATION.")
                    else:
                        print("{} ({}) decomposition -> {}".format(at.get_chemical_formula(), pred_e, subs))
                        for s in subs:
                            print("{} - {} ::: {} - {} = {}".format(formula, s, start, atomic_e[s], start - atomic_e[s]))
                            start -= atomic_e[s]
                        print("Predicted Atomization Energy for {} : {}".format(formula, start))
                        print("Reference Atomization Energy for {} : {}".format(formula, at.info['atomization']))
                        print("Error: {}".format(start - at.info['atomization']))

                for prod in products:
                    reactants = [i for i in val if i.info['product'] == prod]
                    energies = [i.info['enmult']*val_e[i.get_chemical_formula()] for i in bhs]
                    bhr = [i.info['bh_ref'] for i in reactants if i.info.get('bh_ref', None)][0]
                    print("VALIDATION: BH -- {}: {}".format(prod, reactants))
                    bh = sum(energies)
                    print("CALCULATED BH: {}".format(bh))
                    print("REFERENCE BH: {}".format(bhr))
                    print("ERROR: |{} - {}| = {}".format(bh, bhr, abs(bh-bhr)))

        if chkpt_idx > args.chkptmax:
            print("Max checkpoint number reached -- aborting training process.")
            break

                
                    



