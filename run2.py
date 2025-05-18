"""Run experiments with customized loops."""
import os
import copy
from main import main
from configs.point import hyperparameter_dict_default
from exp import Exp

hyperparameter_dict_default['saving']['result_dir']='results'
hyperparameter_dict_default['training']['model_load_path']=r"model_weights\scorenet_20_0.01_8.pth"
exp = Exp(main, hyperparameter_dict_default)

hyperparameter_dict=copy.copy(hyperparameter_dict_default)
#hyperparameter_dict['sampling']['k_i']=0.1
#exp.run(hyperparameter_dict)
#exp.line_run('sampling.k_d', [10.0,11.0], fixed_kv_pairs=[('sampling.k_d_decay', 0.9), ('sampling.k_i', 0.1), ('sampling.k_i_decay', 0.9)])
exp.grid_run('sampling.k_d', [0.0, 2.0, 4.0, 6.0, 8.0, 10.0], 'sampling.k_i', [0.0, 0.1, 0.2, 0.3, 0.5], fixed_kv_pairs=[('sampling.k_i_decay', 1.0)])

print('Done.')
