import json
import munch
import os
import sys
import logging
import importlib as imp
import pdb

import numpy as np
import tensorflow as tf
from absl import app
from absl import flags
from absl import logging as absl_logging

import MixGANSde3 as Mymodel
import Evaulation
imp.reload(Mymodel)
imp.reload(Evaulation)

os.chdir(sys.path[0])

### Now support:
### 1. Ex1GeoBrownian                    ./configs/Ex1GeoBrownian.json
### 2. Ex2Brownian                       ./configs/Ex2Brownian.json
### 3. Ex3OU                             ./configs/Ex3OU.json
### 4. Ex4ExpDiff                        ./configs/Ex4ExpdiffMix.json
### 5. Ex5TrigMix						 ./configs/Ex5TrigMix.json
### 6. Ex6ExpOU 						 ./configs/Ex6ExpOUMix.json
### 7. Ex7MdOU 						 	 ./configs/Ex7MdOUMix.json
### 8. Ex8DW 						     ./configs/Ex8DWMix.json
### 9. Ex9Expdis 					     ./configs/Ex9ExpdisMix.json
### 10. Ex10SO    					     ./configs/Ex10SOMix.json

### 1. SPDE_Ex1SHeatEqu    				 ./configs/SPDE_Ex1SHeatEqu.json

flags.DEFINE_string('test_name',   'Ex30GBMDecayNoiseMix',                'Name of test')
flags.DEFINE_string('config_path', './configs/Ex30GBMDecayNoiseMix.json', 'Path of config file')
flags.DEFINE_string('model',       None,                        'Name of model')
FLAGS = flags.FLAGS

def Model_select(model_name):
	if model_name=='WGAN-GP3':
		return Mymodel.MixWGANGPSde,Mymodel
	elif model_name==None:
		raise AttributeError("Model_select: please type in model name")
	else:
		raise AttributeError("Model_select: %s model is not supported"%(model_name))

def main(argv):
	del argv
	### Setup path
	result_path = './results'
	root_path = result_path+'/'+FLAGS.test_name
	bestm_path = root_path + '/Best_model/'
	model_path = root_path + '/Test_model/'
	histy_path = root_path + '/Test_history.json'
	predt_path = root_path + '/predict.mat'
	setting_path = root_path + '/Test_config.json'
	Monitor_path = root_path + '/Monitor/'
	if not os.path.exists(root_path):
		os.makedirs(root_path)
	### Load configration
	with open(FLAGS.config_path) as json_data_file:
		config = json.load(json_data_file)
	config = munch.munchify(config)
	config.net_config.model_name = FLAGS.model
	json.dump(config, open(setting_path, 'w'), indent=2)
	### Setup logging information
	absl_logging.get_absl_handler().setFormatter(logging.Formatter('%(asctime)s\t%(levelname)-10s %(message)s'))
	absl_logging.get_absl_handler().use_absl_log_file('Test', root_path)
	absl_logging.set_verbosity('info')
	logging.info('Begin to learn %s ' % config.eqn_config.eqn_name)
	### Model set & Monitor & Evaluation
	MyGansmodel,Mymodule = Model_select(FLAGS.model)
	Eva = Evaulation.SdeGanEva(config,root_path,Monitor_path+'Eva')
	GanMonitor = Mymodule.Monitor(Monitor_path,config,MyGansmodel,Eva)
	### Data Manuplation
	logging.info('Start to manupulate data')
	DatVes = Mymodule.DataTran(config,GanMonitor)
	DatVes.read_traindata()
	DatVes.train_data_trans(0)
	DatVes.read_testdata()
	DatVes.train_hiddendata()
	### Model definition and trained
	logging.info('Start to train %s model'%(FLAGS.model))
	GanModel = MyGansmodel(config)
	GanModel.Model_drift = DatVes.Model_drift
	GanModel.train(DatVes.train_mat,DatVes.train_hiddendata,model_path,histy_path,GanMonitor,DatVes,predt_path)
	### Test model
	##  This part has been transferred into GanModel.train function for general purpose
	# logging.info('Start to test model')
	# DatVes.test_mdat1model(GanModel,predt_path)
	# logging.info('End of test')


if __name__ == '__main__':
	app.run(main)
