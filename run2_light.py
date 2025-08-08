"""Run experiments with customized loops."""
import os
import yaml
import copy
from main_light import main
from exp import Exp

with open(os.path.join('configs', 'point.yml'), 'r') as f:
    hyperparameter_dict_default = yaml.load(f, Loader=yaml.FullLoader)

hyperparameter_dict_default['saving']['result_dir']='results112'
hyperparameter_dict_default['training']['model_load_path']=r"model_weights\scorenet_20_0.01_8.pth"
exp = Exp(main, hyperparameter_dict_default)

hyperparameter_dict=copy.copy(hyperparameter_dict_default)
hyperparameter_dict['logging']['sampling_log_freq']=100
hyperparameter_dict['sampling']['k_p']=1.5
hyperparameter_dict['seed']=1342
#exp.run(hyperparameter_dict)
exp.line_run('sampling.k_d', [0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0])
#exp.grid_run('sampling.k_d', [0.0, 2.0, 4.0, 6.0, 8.0, 10.0], 'sampling.k_i', [0.0, 0.1, 0.2, 0.3, 0.5], fixed_kv_pairs=[('sampling.k_i_decay', 1.0)])

print('Done.')
