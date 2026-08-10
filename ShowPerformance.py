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

import Evaulation
imp.reload(Evaulation)

os.chdir(sys.path[0])

### Now support:
### 1. Ex1GeoBrownian
### 2. Ex2Brownian
### 3. Ex3OU
### 4. Ex4ExpDiff
### 5. Ex5TrigMix
### 5. Ex6ExpOU

flags.DEFINE_string('test_name',   'Ex3OU',                'Name of test')
flags.DEFINE_string('test_case',    None,                        'Name of post test')
FLAGS = flags.FLAGS


def main(argv):
	del argv
	### Setup path
	result_path = './results'
	root_path = result_path+'/'+FLAGS.test_name+'/'+FLAGS.test_case if FLAGS.test_case else result_path+'/'+FLAGS.test_name
	save_path = root_path+'/'+'Eva'
	config_path = root_path+'/'+'Test_config.json'
	if not os.path.exists(save_path):
		os.makedirs(save_path)
	### Load configration
	with open(config_path) as json_data_file:
		config = json.load(json_data_file)
	config = munch.munchify(config)
	### Evaluation
	Eva = Evaulation.SdeGanEva(config,root_path,save_path)
	showcf = config.show_config
	if ('plot_samplecompare' in showcf.keys()) and (showcf.plot_samplecompare):
		Eva.plot_samplecompare(save=True)
	if ('plot_meancompare' in showcf.keys()) and (showcf.plot_meancompare):
		Eva.plot_meancompare(save=True)
	if ('plot_losthist' in showcf.keys()) and (showcf.plot_losthist):
		Eva.plot_losthist(save=True)
		Eva.plot_Wdistance(save=True)


if __name__ == '__main__':
	app.run(main)