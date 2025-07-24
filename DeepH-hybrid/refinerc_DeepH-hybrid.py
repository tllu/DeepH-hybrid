# Script for additionally preprocess for DeepH-hybrid
# Coded by ZC Tang @ Tsinghua Univ for DeepH-hybrid support. e-mail: az_txycha@126.com
# Please follow the instructions in "README" to use this script

import os
import h5py
import json
import argparse
import numpy as np
from tqdm import tqdm
from pathlib import Path
from pathos.multiprocessing import ProcessingPool as Pool


Bohr2Ang = 0.52918

def modify_h5(h5_in, h5_out, Rxlist, Rylist, Rzlist, max_rc, all_atoms, lat, element_rc, element_info, nao):
    dist = np.zeros(len(all_atoms))
    if h5_in == h5_out:
        print("Please don't try overwriting {} for safety concern!".fromat(h5_in))
        exit()
    with h5py.File(h5_in, 'r') as S_original_f:
        with h5py.File(h5_out, 'w', libver='latest') as S_new_f:
            S_new = {}
            for Rx in Rxlist:
                for Ry in Rylist:
                    for Rz in Rzlist:
                        mirror_atoms = all_atoms.copy()
                        mirror_atoms += (Rx * lat[0,:])[None, :] + (Ry * lat[1,:])[None, :] + (Rz * lat[2,:])[None, :]
                        for ia in range(len(all_atoms)):
                            dist[:] = np.linalg.norm((mirror_atoms-(all_atoms[ia,:])[None,:]),axis=1)
                            if len(np.where(dist<max_rc * Bohr2Ang * 2)[0]) > 0:
                                for ja in np.where(dist<max_rc * Bohr2Ang * 2)[0]:
                                    this_key = "[{}, {}, {}, {}, {}]".format(Rx,Ry,Rz,ia+1,ja+1)
                                    t_rc = (element_rc[element_info[ia]] + element_rc[element_info[ja]]) * Bohr2Ang
                                    if dist[ja] > t_rc:
                                        continue
                                    if this_key not in S_original_f.keys():
                                        S_new[this_key] = np.zeros((nao[ia+1],nao[ja+1]))
                                    else:
                                        S_new[this_key] = np.array(S_original_f[this_key],dtype=np.float32)
            for key,value in S_new.items():
                S_new_f[key] = value


def process(work_dir, element_rc, only_S):
    # generate Rxlist, Rylist, Rzlist according to reciprocal cell and maximal cutoff
    
    work_dir = Path(work_dir)
    
    element_info = np.loadtxt( work_dir / "element.dat", ndmin=1 )
    
    _element_info = list( set(element_info) )
    max_rc = element_rc[_element_info[0]]
    for el_number in _element_info[1:]:
        if element_rc[el_number] > max_rc:
            max_rc = element_rc[el_number]
    
    max_rc = max(element_rc.values())
    rlat = np.transpose(np.loadtxt(  work_dir / "rlat.dat" ))
    nRx = (int(np.ceil(max_rc*Bohr2Ang/np.pi*np.linalg.norm(rlat[0,:]))) +1) * 2 - 1
    nRy = (int(np.ceil(max_rc*Bohr2Ang/np.pi*np.linalg.norm(rlat[1,:]))) +1) * 2 - 1
    nRz = (int(np.ceil(max_rc*Bohr2Ang/np.pi*np.linalg.norm(rlat[2,:]))) +1) * 2 - 1
    Rxlist = np.arange(nRx)-int((nRx-1)/2)
    Rylist = np.arange(nRy)-int((nRy-1)/2)
    Rzlist = np.arange(nRz)-int((nRz-1)/2)

    all_atoms = np.transpose(np.loadtxt( work_dir / "site_positions.dat", ndmin=2 ))
    nao = {} # orbital number of every site
    
    with open( work_dir / "orbital_types.dat", 'r') as ot_f:
        this_ia = 1
        line = ot_f.readline()
        while line:
            line = line.strip().split()
            this_nao = 0
            for itype in range(len(line)):
                this_nao += int(line[itype]) * 2 + 1
            nao[this_ia] = this_nao
            this_ia += 1
            line = ot_f.readline()
    lat = np.transpose(np.loadtxt( work_dir / "lat.dat" ))

    overlaps_in_path = work_dir / "overlaps.h5"
    overlaps_refined_path = work_dir / "overlaps_refined.h5"
    modify_h5(
        overlaps_in_path,
        overlaps_refined_path,
        Rxlist,
        Rylist, 
        Rzlist, 
        max_rc, 
        all_atoms, 
        lat, 
        element_rc, 
        element_info, 
        nao
    )
    os.rename(overlaps_refined_path, overlaps_in_path)
    
    if not only_S:
        hamiltonians_in_path = work_dir / "hamiltonians.h5"
        hamiltonians_refined_path = work_dir / "hamiltonians_refined.h5"
        modify_h5(
            hamiltonians_in_path,
            hamiltonians_refined_path, 
            Rxlist, 
            Rylist, 
            Rzlist, 
            max_rc, 
            all_atoms, 
            lat, 
            element_rc, 
            element_info, 
            nao
        )
        os.rename(hamiltonians_refined_path, hamiltonians_in_path)


def modify_DeepH_hybrid(input_path, element_rc, only_S, multiprocess, n_jobs):    
    input_path = Path(input_path)
    has_subdir = False
    for work_dir in input_path.iterdir():
        if work_dir.is_dir():
            has_subdir = True
            break

    if has_subdir:
        if multiprocess:
            pool_dict = {'nodes': n_jobs}
            print(f"Using multiprocess with {n_jobs} jobs.")
            with Pool(**pool_dict) as pool:            
                files_to_process = list(input_path.iterdir())
                total_files = len(files_to_process)
                
                for _ in tqdm(
                    pool.uimap(lambda x: process(x, element_rc, only_S), files_to_process), 
                    total=total_files
                ):
                    pass
        else:
            for work_dir in tqdm(list(input_path.iterdir())):
                process(work_dir, element_rc, only_S)
    else:
        process(input_path, element_rc, only_S)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Modifications for DeepH-hybrid')
    parser.add_argument(
        '-i','--input_dir', type=str, default='./',
        help='path of directory to be modifyed'
        )
    parser.add_argument(
        '-c','--config', type=str, default='rc_config.json',
        help='path of the config file for modification of the cut-off radius'
        )
    parser.add_argument(
        '-S','--only_S', type=int, default=0
        )
    parser.add_argument(
        '-mp', '--multiprocess', action='store_true',
        help='use multiprocess to speed up the process, default is False',
    )
    parser.add_argument(
        '-j', '--n_jobs', type=int, default=1,
        help='number of jobs to run in parallel, default is 1',
    )
    args = parser.parse_args()

    input_path = args.input_dir
    config_path = args.config
    only_S = bool(args.only_S)
    multiprocess = args.multiprocess
    n_jobs = args.n_jobs

    with open(config_path, 'r') as config_f:
        element_rc_raw = json.load(config_f)
        element_rc = {}
        for (key, value) in element_rc_raw.items():
            element_rc[int(key)] = value
    
    modify_DeepH_hybrid(input_path, element_rc, only_S, multiprocess, n_jobs)