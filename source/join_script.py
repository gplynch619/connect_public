import sys
from default_module import Parameters
import os
from tools import join_data_files


CONNECT_PATH  = os.path.realpath(os.path.dirname(__file__))
param_file    = sys.argv[1]
param         = Parameters(param_file)
path = CONNECT_PATH + f'/data/{param.jobname}/'

join_data_files(param)