import json
import munch
import os
import sys
import logging
import importlib as imp

import numpy as np
import tensorflow as tf
from absl import app
from absl import flags
from absl import logging as absl_logging
import pdb

import MixGANSde3 as Mymodel
import Evaulation
imp.reload(Mymodel)
imp.reload(Evaulation)

os.chdir(sys.path[0])

### Now support:
### 1. Ex1GeoBrownian
### 2. Ex2Brownian
### 3. Ex3OU

flags.DEFINE_string('test_name',   'Excute_Ex8s05DW_3s3',                'Name of test')
flags.DEFINE_string('test_case',    None,                        'Name of post test')
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
	### Model name
	MODELNAME = 'WGAN-GP3'
	### Setup path
	result_path = './results'
	root_path = result_path+'/'+FLAGS.test_name+'/'+FLAGS.test_case if FLAGS.test_case else result_path+'/'+FLAGS.test_name
	save_path = root_path+'/'+'PostEva/'
	config_path = root_path+'/'+'Test_config.json'
	if not os.path.exists(save_path):
		os.makedirs(save_path)
	### Load configration
	with open(config_path) as json_data_file:
		config = json.load(json_data_file)
	config = munch.munchify(config)
	### Evaluation
	MyGansmodel,Mymodule = Model_select(MODELNAME)
	Eva_M = Evaulation.SdeGanEva(config,root_path,save_path+'Eva')
	Eva   = Evaulation.SdeGanEva(config,root_path,root_path+'/Eva')
	GanMonitor = Mymodule.Monitor(save_path,config,MyGansmodel,Eva_M)
	### Data Manuplation
	DatVes = Mymodule.DataTran(config,GanMonitor)
	DatVes.read_traindata()
	DatVes.train_data_trans(0)
	DatVes.read_testdata()
	DatVes.train_hiddendata()
	### Model definition and trained
	GanModel = Evaulation.Evaluate.readmodel(root_path+'/Test_model',MyGansmodel,config)
	GanModel.Model_drift = DatVes.Model_drift
	### Validation
	checkpoint_Ens = tf.train.Checkpoint(G_optimizer=GanModel.G_optimizer,D_optimizer=GanModel.D_optimizer,G=GanModel.G,D=GanModel.D)
	ckptmanager_Ens = tf.train.CheckpointManager(checkpoint_Ens, root_path+'/Monitor/Ens_model/', max_to_keep=10)
	# Eva.plot_meancompare(save=True)
	# Eva.plot_samplecompare(save=True)
	GanMonitor.Ens_predictor(0,ckptmanager_Ens,checkpoint_Ens,GanModel,DatVes,None,root_path+'/a')
	


if __name__ == '__main__':
	app.run(main)
