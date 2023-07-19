import sys
from default_module import Parameters
import os
from source.join_output import CreateSingleDataFile

CONNECT_PATH  = os.path.realpath(os.path.dirname(__file__))
param_file    = sys.argv[1]
param         = Parameters(param_file)
path = CONNECT_PATH + f'/data/{param.jobname}/'

CSDF = CreateSingleDataFile(param, CONNECT_PATH)
CSDF.join()