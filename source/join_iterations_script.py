import sys
from default_module import Parameters
import os
from tools import combine_sets_of_data_files

CONNECT_PATH  = os.path.realpath(os.path.dirname(__file__))
param_file    = sys.argv[1]
old_data      = sys.argv[2]
new_data      = sys.argv[3]

param         = Parameters(param_file)

path = CONNECT_PATH + f'/data/{param.jobname}/'
old_data_path = sys.path.join(path, old_data)
new_data_path = sys.path.join(path, new_data)

combine_sets_of_data_files(os.path.join(new_data_path, 'model_params.txt'),
                               os.path.join(old_data_path, 'model_params.txt'))
if len(param.output_derived) > 0:
    combine_sets_of_data_files(os.path.join(new_data_path, 'derived.txt'),
                                os.path.join(old_data_path, 'derived.txt'))
for output in param.output_Cl:
    combine_sets_of_data_files(os.path.join(new_data_path, f'Cl_{output}.txt'),
                                os.path.join(old_data_path, f'Cl_{output}.txt'))
for output in param.output_Pk:
    combine_sets_of_data_files(os.path.join(new_data_path, f'Pk_{output}.txt'),
                                os.path.join(old_data_path, f'Pk_{output}.txt'))
for output in param.output_z:
    combine_sets_of_data_files(os.path.join(new_data_path, f'{output}_z.txt'),
                                os.path.join(old_data_path, f'{output}_z.txt'))
for output in param.output_bg:
    combine_sets_of_data_files(os.path.join(new_data_path, f'Cl_{output}.txt'),
                                os.path.join(old_data_path, f'Cl_{output}.txt'))
for output in param.output_th:
    combine_sets_of_data_files(os.path.join(new_data_path, f'{output}.txt'),
                                os.path.join(old_data_path, f'{output}.txt'))