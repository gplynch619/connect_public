from importlib.machinery import SourceFileLoader
import numpy as np


class Parameters():
    def __init__(self, param_file):
        param = SourceFileLoader(param_file, param_file).load_module()
        jobname = param_file.split('/')[-1].split('.')[0]

        self.param_file = param_file
        self.parameters = param.parameters

        self.jobname              = getattr(param, 'jobname',              jobname             )
        self.N                    = getattr(param, 'N',                    10000               )
        self.train_ratio          = getattr(param, 'train_ratio',          0.9                 )
        self.val_ratio            = getattr(param, 'val_ratio',            0.01                )
        self.save_name            = getattr(param, 'save_name',            None                )
        self.overwrite_model      = getattr(param, 'overwrite_model',      False               )
        self.batchsize            = getattr(param, 'batchsize',            64                  )
        self.epochs               = getattr(param, 'epochs',               200                 )
        self.loss_function        = getattr(param, 'loss_function',        'cosmic_variance'   )
        self.activation_function  = getattr(param, 'activation_function',  'alsing'            )
        self.N_nodes              = getattr(param, 'N_nodes',              512                 )
        self.N_hidden_layers      = getattr(param, 'N_hidden_layers',      4                   )
        self.output_activation    = getattr(param, 'output_activation',    False               )
        self.extra_input          = getattr(param, 'extra_input',          {}                  )
        self.output_Cl            = getattr(param, 'output_Cl',            ['tt']              )
        self.ll_max_request       = getattr(param, 'll_max_request',       2500                )
        self.output_z             = getattr(param, "output_z",             []                  )
        self.output_z_grids       = getattr(param, "output_z_grids",       self.get_z_grids()  )
        self.output_Pk            = getattr(param, 'output_Pk',            []                  )
        self.output_Pk_grid       = getattr(param, 'output_Pk_grid',       self.get_kz_array() )
        self.output_bg            = getattr(param, 'output_bg',            []                  )
        self.output_th            = getattr(param, 'output_th',            []                  )
        self.output_derived       = getattr(param, 'output_derived',       []                  )
        self.N_max_points         = getattr(param, 'N_max_points',         10000               )
        self.normalization_method = getattr(param, 'normalization_method', 'standardization'   )
        self.sampling             = getattr(param, 'sampling',             'lhc'               )
        self.input_model_file     = getattr(param, 'input_model_file',     None                )
        self.mcmc_sampler         = getattr(param, 'mcmc_sampler',         'cobaya'            )
        self.initial_model        = getattr(param, 'initial_model',        None                )
        self.resume_iterations    = getattr(param, 'resume_iterations',    False               )
        self.control_point_priors = getattr(param, 'control_point_priors', None                )
        self.prior_ranges         = getattr(param, 'prior_ranges',         {}                  )
        self.log_priors           = getattr(param, 'log_priors',           []                  )
        self.bestfit_guesses      = getattr(param, 'bestfit_guesses',      {}                  )
        self.sigma_guesses        = getattr(param, 'sigma_guesses',        {}                  )
        self.keep_first_iteration = getattr(param, 'keep_first_iteration', False               )
        self.mcmc_tol             = getattr(param, 'mcmc_tol',             0.01                )
        self.iter_tol             = getattr(param, 'iter_tol',             0.01                )
        self.sampling_likelihoods = getattr(param, 'sampling_likelihoods', ['Planck_lite']     )
        self.extra_cobaya_lkls    = getattr(param, 'extra_cobaya_lkls',    {}                  ) 

    def get_kz_array(self):
        if len(self.output_Pk) > 0:
            z_vec = np.logspace(np.log(1e-2),
                                np.log(20),
                                10)
            z_vec = np.insert(z_vec,
                              0, 
                              0.0)
            k_vec = np.array(
                sorted(
                    list(np.logspace(np.log(0.04),
                                     np.log(0.5),
                                     30,base=np.exp(1.0)))
                    +
                    [x for x in np.array(
                        [x if x < 0.04 or x > 0.5 else None
                         for x in np.logspace(np.log(1e-3),
                                              np.log(1.0),
                                              20,
                                              base=np.exp(1.0))]) if x is not None]))
            return [k_vec, z_vec]
        else:
            return None
        
    def get_z_grids(self):
        grids = {}
        if "H" in self.output_z:
            z = np.logspace(-2, np.log10(5), 49)
            z = np.insert(z, 0, 0.0)
            grids["H"] = z
        if "angular_distance" in self.output_z:
            z = np.logspace(-2, np.log10(5), 49)
            z = np.insert(z, 0, 0.0)
            grids["angular_distance"] = z
        if "sigma8_z" in self.output_z:
            z = np.logspace(-2, np.log10(5), 49)
            z = np.insert(z, 0, 0.0)
            grids["sigma8_z"] = z
        if "x_e" in self.output_z:
            grids["x_e"] = np.linspace(700, 1700, 100)
        if "g" in self.output_z:
            grids["g"] = np.linspace(600, 1500, 100)
        
        return grids
