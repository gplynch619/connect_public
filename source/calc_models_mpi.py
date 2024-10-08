import os
import sys
import time
import itertools
import pickle as pkl
import signal
import traceback
#sys.path.insert(0, '/home/gplynch/projects/connect_public')
import datetime
sys.path.insert(0, "/home/gplynch/projects/emu_wrapper")
from TrainedEmulator import *

os.environ['UCX_LOG_LEVEL'] = 'error'
param_file   = sys.argv[1]
CONNECT_PATH = sys.argv[2]
sampling     = sys.argv[3]
sys.path.insert(0,CONNECT_PATH)

import numpy as np
import classy
from scipy.interpolate import CubicSpline, interp1d
from scipy.special import logit
from mpi4py import MPI

from source.default_module import Parameters
from source.tools import get_computed_cls, get_z_idx, get_covmat
#from source.ini_samplers import *

param_file = os.path.join(CONNECT_PATH, param_file)
param        = Parameters(param_file)
param_names  = list(param.parameters.keys())

path = os.path.join(CONNECT_PATH, f'data/{param.jobname}')
if sampling == 'iterative':
    try:
        iteration = max([int(f.split('number_')[-1]) for f in os.listdir(path) if f.startswith('number')])
        directory = os.path.join(path, f'number_{iteration}')
    except:
        directory = os.path.join(path, f'N-{param.N}')
elif sampling in ['lhc','hypersphere','pickle']:
    directory = os.path.join(path, f'N-{param.N}')
elif sampling == "recompute":
    directory = os.path.join(path, "number_0")
    model_progress_file = os.path.join(directory, "model_progress.txt")

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
N_slaves = comm.Get_size()-1
get_slave = itertools.cycle(range(1,N_slaves+1))

def excepthook(etype, value, tb):
    if tb is not None and value not in [None, '']:
        print('Traceback (most recent call last):', file=sys.stderr)
        traceback.print_tb(tb, file=sys.stderr)
        print(f'{etype.__name__}: {value.args[0]}', file=sys.stderr)
    comm.Abort(1)
sys.excepthook = excepthook



if len(param.output_Cl) > 0:
    cosmo = classy.Class()
    input_params = {'output':'tCl, lCl, pCl', 'lensing':'yes'}
    if 'l_max_scalars' in param.extra_input:
        input_params.update({'l_max_scalars':param.extra_input['l_max_scalars']})
    cosmo.set(input_params)
    cosmo.compute()
    cls = get_computed_cls(cosmo, ll_max_request=param.ll_max_request)
    global_ell = cls['ell']
    

## rank == 0 (master)
if rank == 0:
    if sampling == 'iterative':
        print("IN ITERATIVE BRANCH")
        try:
            exec(f'from source.mcmc_samplers.{param.mcmc_sampler} import {param.mcmc_sampler}')
        except Exception as e:
            print(e)
            raise
        _locals = {}
        print(CONNECT_PATH)
        exec(f'mcmc = {param.mcmc_sampler}(param, CONNECT_PATH)', locals(), _locals)
        mcmc = _locals['mcmc']
        print("ABOUT TO IMPORT")

        data = mcmc.import_points_from_chains(iteration)
        print("END ITERATIVE BRANCH")


        
    elif sampling == 'lhc':
        from source.ini_samplers import LatinHypercubeSampler
        lhc = LatinHypercubeSampler(param)
        data = lhc.run()

    elif sampling == 'hypersphere':
        from source.ini_samplers import HypersphereSampler
        hs = HypersphereSampler(param)
        data = hs.run()

    elif sampling == 'pickle':
        from source.ini_samplers import PickleSampler
        ps = PickleSampler(param)
        data = ps.run()

    print(data.shape)
        
    sleep_short = 0.0001
    sleep_long = 0.1
    sleep_dict = {}
    for r in range(1,N_slaves+1):
        sleep_dict[r] = 1

    data_idx = 0
    while len(data) > data_idx:
        r = next(get_slave)
        if comm.iprobe(r):
            last_model_id = int(comm.recv(source=r))
            model = np.insert(data[data_idx], 0, data_idx)
            comm.send(model, dest=r)
            if sampling=="recompute":
                if last_model_id!=-1:
                    with open(model_progress_file, "a") as f:
                        f.write(f"{remaining_idx[last_model_id]}\n")
            data_idx += 1
            sleep_dict[r] = 1
        else:
            sleep_dict[r] = 0

        if all(value == 0 for value in sleep_dict.values()):
            time.sleep(sleep_long)
        else:
            time.sleep(sleep_short)

    for r in range(1,N_slaves+1):
        comm.send('Done', dest=r)



## rank > 0 (slaves)
else:

    xe_fid_max=1.09258
    # Directories for input (model parameters) and output (Cl data) data
    def inverse_pert_transformation(qt,xe_fid,xe_max=xe_fid_max):
        shift = logit(xe_fid/xe_max)
        res = logit( (qt+xe_fid)/xe_max ) - shift
        if np.isnan(res):
            print("NaN from qt {} xe_fid {}".format(qt, xe_fid))
        if np.isinf(res):
            print("inf from qt {} xe_fid {}".format(qt, xe_fid))
        return res

    if "xe_control_pivots" in param.extra_input.keys():
        pivots = np.array([float(z) for z in param.extra_input["xe_control_pivots"].split(",")])
    
    if np.array([p.startswith("qt") for p in param.parameters.keys()]).any():
        lcdm_emulator = TrainedEmulator("/home/gplynch/projects/connect_public/trained_models/lcdm_xe")

    # Directories for input (model parameters) and output (Cl data) data
    in_dir           = os.path.join(directory, f'model_params_data/model_params_{rank}.txt')
    out_dirs_Cl      = []
    out_dirs_unlensed_Cl = []
    out_dirs_Pk      = []
    out_dirs_z       = []
    out_dirs_bg      = []
    out_dirs_th      = []
    out_dirs_ex      = []
    for Cl in param.output_Cl:
        out_dirs_Cl.append(os.path.join(directory, f'Cl_{Cl}_data/Cl_{Cl}_data_{rank}.txt'))
    for Cl in param.output_unlensed_Cl:
        out_dirs_unlensed_Cl.append(os.path.join(directory, f'Cl_unlensed_{Cl}_data/Cl_unlensed_{Cl}_data_{rank}.txt'))
    for Pk in param.output_Pk:
        out_dirs_Pk.append(os.path.join(directory, f'Pk_{Pk}_data/Pk_{Pk}_data_{rank}.txt'))
    for bg in param.output_bg:
        bg = bg.replace('/','\\')
        out_dirs_bg.append(os.path.join(directory, f'bg_{bg}_data/{bg}_data_{rank}.txt'))
    for th in param.output_th:
        out_dirs_th.append(os.path.join(directory, f'th_{th}_data/{th}_data_{rank}.txt'))
    if len(param.output_derived) > 0:
        out_dir_derived = os.path.join(directory, f'derived_data/derived_data_{rank}.txt')
    for ex in param.extra_output:
        out_dirs_ex.append(os.path.join(directory, f'extra_{ex}_data/{ex}_data_{rank}.txt'))

    param_header = '# '
    for par_name in param_names:
        if par_name == param_names[-1]:
            param_header += par_name+'\n'
        else:
            param_header += par_name+'\t'

    derived_header = '# '
    for der_name in param.output_derived:
        if der_name == param.output_derived[-1]:
            derived_header += der_name+'\n'
        else:
            derived_header += der_name+'\t'

    # Initialise data files
    with open(in_dir, 'w') as f:
        f.write(param_header)

    for out_dir in out_dirs_Cl + out_dirs_unlensed_Cl + out_dirs_Pk + out_dirs_bg + out_dirs_th:
        with open(out_dir, 'w') as f:
            f.write('')
    try:
        with open(out_dir_derived, 'w') as f:
            f.write(derived_header)
    except:
        pass

    # Initialise timeout signal
    # def timeout_handler(num, stack):
    #     raise Exception('timeout')
    # signal.signal(signal.SIGALRM, timeout_handler)
    

    # Iterate over each model
    last_model_idx = -1
    while True:
        comm.send(last_model_idx, dest=0)
        model = comm.recv(source=0)
        if type(model).__name__ == 'str':
            break
        last_model_idx = model[0]
        model = model[1:]
        #print("RANK {} received model {}".format(rank, model))
        #s=""
        #for p in model:
        #    s+="{} ".format(p)
        #with open("logs/rank_{}.log".format(rank), "a+") as f:
        #    f.write("{}\n".format(s))
        # Set required CLASS parameters
        params = {}
        if len(param.output_Cl) > 0:
            params['output']            = 'tCl,lCl'
            params['lensing']           = 'yes'
            if any("b" in s or "e" in s for s in param.output_Cl):
                params['output']       += ',pCl'

        params.update(param.extra_input)
 
        for i, par_name in enumerate(param_names):
            if par_name.startswith("q_"):
                continue
            elif par_name.startswith("qt_"):
                continue
            else:
                params[par_name] = model[i]
        if param.extra_input["xe_pert_type"]=="control":
            if param.extra_input["xe_interp_type"]!="linear":
                if np.array([p.startswith("qt") for p in param.parameters.keys()]).any():
                    cosmo_params = [params[p] for p in param_names if (not p.startswith("q") and not p.startswith("w0"))]
                    fid_xe = lcdm_emulator.get_predictions([cosmo_params])[0]
                    fid_xe_func = interp1d(lcdm_emulator.output_info["output_z_grids"]["x_e"], fid_xe)
        value_error_flag = False
        try:
            if param.extra_input["xe_pert_type"]=="control":
                model_cps = [0.0]
                for i, par_name in enumerate(param_names):
                    if par_name.startswith("q_"):
                        model_cps.append(model[i])
                    elif par_name.startswith("qt_"):
                        if param.extra_input["xe_interp_type"]=="linear":
                            model_cps.append(model[i])
                        else:
                            qid = int(par_name.split("_")[1])
                            fid_xe_at_zi = fid_xe_func(pivots[qid])
                            if model[i]<(-fid_xe_at_zi):
                                model[i]=-0.999*fid_xe_at_zi
                            elif model[i]>(xe_fid_max-fid_xe_at_zi):
                                model[i]=0.999*(xe_fid_max-fid_xe_at_zi)
                            qi = inverse_pert_transformation(model[i],fid_xe_at_zi)
                            if np.isnan(qi):
                                raise ValueError
                            elif np.isinf(qi):
                                raise ValueError
                            model_cps.append(qi)
                model_cps.append(0.0) # first and last control points fixed to zero 
                string_cps = ["{:.5e}".format(c) for c in model_cps] #8 decimal places
                cp_string = ",".join(string_cps)
                params["xe_control_points"] = cp_string
        except KeyError:
            continue
        except ValueError:
            value_error_flag = True


        if 'P_k_max_h/Mpc' in params:
            val = params.pop('P_k_max_h/Mpc')
            params['P_k_max_1/Mpc'] = val*0.67556
        if len(param.output_Pk) > 0:
            if 'output' in params:
                params['output']       += ',mPk'
            else:
                params['output']        = 'mPk'
            params['P_k_max_1/Mpc']     = 2.5*max(param.k_grid)
            params['z_max_pk']          = max(param.z_Pk_list)

        if 'sigma8' in param.output_derived:
            if not 'mPk' in params['output']:
                if len(params['output']) > 0:
                    params['output']   += ',mPk'
                else:
                    params['output']    = 'mPk'
            if not 'P_k_max_1/Mpc' in params:
                params['P_k_max_1/Mpc'] = 1.

        if "effective_f_sigma8" in param.extra_output:
                params['z_max_pk']          = max(param.z_bg_list)+.1
                params['P_k_max_1/Mpc'] = 1.

        # signal.alarm(200) # CLASS computations must not take longer than 200 seconds
        try:
            cosmo = classy.Class()
            cosmo.set(params)
            cosmo.compute()
            if len(param.output_bg) > 0:
                bg = cosmo.get_background()
                if len(param.z_bg_list) > 0:
                    z_bg = param.z_bg_list
                else:
                    bg_idx = get_z_idx(bg['z'])
                    z_bg = bg['z'][bg_idx]
            if len(param.output_th) > 0:
                th = cosmo.get_thermodynamics()
                if len(param.z_th_list) > 0:
                    z_th = param.z_th_list
                else:
                    th_idx = get_z_idx(th['z'])
                    z_th = th['z'][th_idx]
            if len(param.output_derived) > 0:
                der = cosmo.get_current_derived_parameters(param.output_derived)
            if len(param.output_Cl) > 0:
                cls = get_computed_cls(cosmo, ell_array=global_ell)
                if any(np.isnan(cls[key]).any() for key in cls):
                    raise classy.CosmoComputationError(
                        'Class computation completed with NaN values in CMB power spectra.')
                ell = cls['ell'][2:]
            if len(param.output_unlensed_Cl) > 0:
                unlensed_cls = get_computed_cls(cosmo, ell_array=global_ell, lensed=False)
                if any(np.isnan(cls[key]).any() for key in cls):
                    raise classy.CosmoComputationError(
                        'Class computation completed with NaN values in CMB power spectra.')
                ell = cls['ell'][2:]
                unlensed_ell = unlensed_cls['ell'][2:]
            if len(param.output_Pk) > 0:
                pks = {}
                for pk in param.output_Pk:
                    pks[pk] = {}
                    for z in param.z_Pk_list:
                        pks[pk][z] = []
                        for k in param.k_grid:
                            pks[pk][z].append(eval(f'cosmo.{pk}(k,z)'))
            success = True
        except classy.CosmoComputationError as e:
            print('The following model failed in CLASS:', flush=True)
            print(model, flush=True)
            success = False
            print(e.message)
        except classy.CosmoSevereError as e:
            print('The following model failed in CLASS:', flush=True)
            print(model, flush=True)
            success = False
            print(e.message)
        # except Exception as e:
        #     if str(e) == 'timeout':
        #         print('The following model took too long to complete:', flush=True)
        #         print(params, flush=True)
        #         success = False
        #     else:
        #         raise e
        # finally:
        #     signal.alarm(0)
                
        if success:
            # Write data to data files
            for out_dir, output in zip(out_dirs_Cl, param.output_Cl):
                par_out = cls[output][2:]*ell*(ell+1)/(2*np.pi)
                with open(out_dir, 'a') as f:
                    for i, l in enumerate(ell):
                        if i != len(ell)-1:
                            f.write(str(l)+'\t')
                        else:
                            f.write(str(l)+'\n')
                    for i, p in enumerate(par_out):
                        if i != len(par_out)-1:
                            f.write(str(p)+'\t')
                        else:
                            f.write(str(p)+'\n')

            for out_dir, output in zip(out_dirs_unlensed_Cl, param.output_unlensed_Cl):
                par_out = unlensed_cls[output][2:]*unlensed_ell*(unlensed_ell+1)/(2*np.pi)
                with open(out_dir, 'a') as f:
                    for i, l in enumerate(unlensed_ell):
                        if i != len(unlensed_ell)-1:
                            f.write(str(l)+'\t')
                        else:
                            f.write(str(l)+'\n')
                    for i, p in enumerate(par_out):
                        if i != len(par_out)-1:
                            f.write(str(p)+'\t')
                        else:
                            f.write(str(p)+'\n')

            for out_dir, output in zip(out_dirs_Pk, param.output_Pk):
                with open(out_dir, 'a') as f:
                    for i, k in enumerate(param.k_grid):
                        if i != len(param.k_grid)-1:
                            f.write(str(k)+'\t')
                        else:
                            f.write(str(k)+'\n')
                    for z in param.z_Pk_list:
                        par_out = pks[output][z]
                        for i, p in enumerate(par_out):
                            if i != len(par_out)-1:
                                f.write(str(p)+'\t')
                            else:
                                f.write(str(p)+'\n')

            if len(param.output_derived) > 0:
                par_out = []
                for output in param.output_derived:
                    par_out.append(der[output])
                with open(out_dir_derived, 'a') as f:
                    for i, p in enumerate(par_out):
                        if i != len(par_out)-1:
                            f.write(str(p)+'\t')
                        else:
                            f.write(str(p)+'\n')

            for out_dir, output in zip(out_dirs_bg, param.output_bg):
                if len(param.z_bg_list) > 0:
                    par_out = CubicSpline(np.flip(bg['z']), np.flip(bg[output]), bc_type='natural')(param.z_bg_list)
                else:
                    par_out = bg[output][bg_idx]
                with open(out_dir, 'a') as f:
                    for i, z in enumerate(z_bg):
                        if i != len(z_bg)-1:
                            f.write(str(z)+'\t')
                        else:
                            f.write(str(z)+'\n')
                    for i, p in enumerate(par_out):
                        if i != len(par_out)-1:
                            f.write(str(p)+'\t')
                        else:
                            f.write(str(p)+'\n')

            for out_dir, output in zip(out_dirs_th, param.output_th):
                if len(param.z_th_list) > 0:
                    par_out = CubicSpline(th['z'], th[output], bc_type='natural')(param.z_th_list)
                else:
                    par_out = th[output][th_idx]
                with open(out_dir, 'a') as f:
                    for i, z in enumerate(z_th):
                        if i != len(z_th)-1:
                            f.write(str(z)+'\t')
                        else:
                            f.write(str(z)+'\n')
                    for i, p in enumerate(par_out):
                        if i != len(par_out)-1:
                            f.write(str(p)+'\t')
                        else:
                            f.write(str(p)+'\n')
        
            for out_dir, output in zip(out_dirs_ex, param.extra_output):
                par_out = eval(param.extra_output[output])
                try:
                    len(par_out)
                except:
                    par_out = [par_out]
                with open(out_dir, 'a') as f:
                     for i, p in enumerate(par_out):
                        if i != len(par_out)-1:
                            f.write(str(p)+'\t')
                        else:
                            f.write(str(p)+'\n')
        
            with open(in_dir, 'a') as f:
                for i, m in enumerate(model):
                    if i != len(model)-1:
                        f.write(str(m)+'\t')
                    else:
                        f.write(str(m)+'\n')
        else:
            nfailed[0]+=1
        cosmo.struct_cleanup()

comm.Reduce(nfailed, total_failed, MPI.SUM, 0)
if(rank==0):
    print("{0}/{1} models succeeded".format(len(data)-total_failed[0], len(data)))
comm.Barrier()
MPI.Finalize()
sys.exit(0)
