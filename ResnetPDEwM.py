from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import scipy.io as sio
import tensorflow as tf
import ModelCheckpoint as MCp
import cyc_callback as CLr
import json
import pdb
import logging

class ResnetPDEwM(tf.keras.Model):
	def __init__(self,config):
		super().__init__()
		## parameter set
		self.eqn_config = config.eqn_config
		self.net_config = config.net_config
		self.dim = self.eqn_config.dim
		self.N_his        = self.net_config.N_his
		self.d_RNN        = self.net_config.N_rec
		self.n_hidden     = self.net_config.N_hidden
		self.n_nodes      = self.net_config.N_nodes
		self.n_epochs     = self.net_config.N_epochs
		self.batch_size   = self.net_config.batch_size
		self.batch_period = self.net_config.batch_period
		self.lr           = self.net_config.lr
		## initialization
		self.dense_layer_list=[]
		for __ in range(self.n_hidden):
			dense_layer = tf.keras.layers.Dense(self.n_nodes, activation='relu')
			self.dense_layer_list += [dense_layer]
		self.out_layer = tf.keras.layers.Dense(self.dim)
		self.slice_layer_LastTwo = tf.keras.layers.Lambda(lambda x: x[:, -self.dim:])
		self.slice_layer_Two = tf.keras.layers.Lambda(lambda x: x[:, self.dim:])

	def call(self, inputs, training=False):
		x = inputs
		x_Two = self.slice_layer_Two(x)
		x_LastTwo = self.slice_layer_LastTwo(x)
		for k in range(self.n_hidden):
			x = self.dense_layer_list[k](x)
		x = self.out_layer(x)
		x_add_mean = tf.keras.layers.Add()([x,x_LastTwo])
		z = x_add_mean
		# recurrent
		for __ in range(self.d_RNN-1):
			x = tf.keras.layers.Concatenate()([x_Two,x_add_mean])
			x_Two = self.slice_layer_Two(x)
			x_LastTwo = self.slice_layer_LastTwo(x)
			for k in range(self.n_hidden):
				x = self.dense_layer_list[k](x)
			x = self.out_layer(x)
			x_add_mean = tf.keras.layers.Add()([x,x_LastTwo])
			z = tf.keras.layers.Concatenate()([z,x_add_mean])
		outputs = z
		return outputs

	def myevaluate(self,inputs):
		x = inputs
		x_Two = self.slice_layer_Two(x)
		x_LastTwo = self.slice_layer_LastTwo(x)
		for k in range(self.n_hidden):
			x = self.dense_layer_list[k](x)
		x = self.out_layer(x)
		x_add_mean = tf.keras.layers.Add()([x,x_LastTwo])
		z = x_add_mean
		return z

	def myevaluateincrem(self,inputs):
		x = inputs
		for k in range(self.n_hidden):
			x = self.dense_layer_list[k](x)
		z = self.out_layer(x)
		return z

	def train(self, inputs_train, outputs_train, temp_model_path, model_path, hist_path):
		self.build_compile()
		# build model
		self.build(input_shape=(None,self.N_his*self.dim))
		# complie model
		if self.lr['type_']=='value':
			callbacks2 = []
		elif self.lr['type_']=='cyclic':
			lra = self.lr
			callbacks2 = [CLr.CyclicLR(base_lr=lra['base'],max_lr=lra['max'],step_size=lra['stepsize'],mode=lra['mode'],gamma=lra['gamma'])]
		else:
			raise AttributeError('ResnetPDEwM::train: No this type of learning rate')
		callback = MCp.ModelCheckpoint2(temp_model_path, inputs=inputs_train,outputs=outputs_train,nepoch=self.n_epochs,
										monitor='loss', verbose=0, mode='min', period=self.batch_period)
		callbacks2 += [callback]
		self.summary()
		hist = self.fit(inputs_train, outputs_train, epochs=self.n_epochs,
	              batch_size=self.batch_size, callbacks=callbacks2)
		self.save_weights(model_path)
		json.dump(hist.history, open(hist_path, 'w'), indent=2)
		return hist

	def build_compile(self):
		# build model
		self.build(input_shape=(None,self.N_his*self.dim))
		# complie model
		if self.lr['type_']=='value':
			self.compile(optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=self.lr['lrate']), loss='mean_squared_error')
		elif self.lr['type_']=='cyclic':
			self.compile(optimizer='adam', loss='mean_squared_error')
		else:
			raise AttributeError('ResnetPDEwM::train: No this type of learning rate')


class ResnetPDEwMJCP2022(tf.keras.Model):
	def __init__(self,config):
		super().__init__()
		## parameter set
		self.eqn_config = config.eqn_config
		self.net_config = config.net_config
		self.dim = self.eqn_config.dim
		self.N_his        = self.net_config.N_his
		self.d_RNN        = self.net_config.N_rec
		self.n_d          = self.net_config.N_d
		self.J            = self.net_config.J
		self.n_a          = self.net_config.N_a
		self.n_nodes      = self.net_config.N_nodes
		self.n_epochs     = self.net_config.N_epochs
		self.batch_size   = self.net_config.batch_size
		self.batch_period = self.net_config.batch_period
		self.lr           = self.net_config.lr
		## initialization
		# Disassembly block
		self.Disassembly_block=[]
		for __ in range(self.J):
			Network = []
			for __ in range(self.n_d):
				dense_layer = tf.keras.layers.Dense(self.n_nodes, activation='tanh')
				Network += [dense_layer]
			Network += [tf.keras.layers.Dense(self.n_nodes)]
			self.Disassembly_block += [Network]
		# Assembly layer
		self.Assembly_layer=[]
		for __ in range(self.n_a):
			dense_layer = tf.keras.layers.Dense(self.J, activation='tanh')
			self.Assembly_layer += [dense_layer]
		self.Assembly_layer += [tf.keras.layers.Dense(1)]
		# Output layer
		self.out_layer = tf.keras.layers.Dense(self.dim)
		self.slice_layer_LastTwo = tf.keras.layers.Lambda(lambda x: x[:, -self.dim:])
		self.slice_layer_Two = tf.keras.layers.Lambda(lambda x: x[:, self.dim:])

	def call(self, inputs, training=False):
		x = inputs
		x_Two = self.slice_layer_Two(x)
		x_LastTwo = self.slice_layer_LastTwo(x)
		x = self.onestep(x)
		x_add_mean = tf.keras.layers.Add()([x,x_LastTwo])
		z = x_add_mean
		# recurrent
		for __ in range(self.d_RNN-1):
			x = tf.keras.layers.Concatenate()([x_Two,x_add_mean])
			x_Two = self.slice_layer_Two(x)
			x_LastTwo = self.slice_layer_LastTwo(x)
			x = self.onestep(x)
			x_add_mean = tf.keras.layers.Add()([x,x_LastTwo])
			z = tf.keras.layers.Concatenate()([z,x_add_mean])
		outputs = z
		return outputs

	def onestep(self, x):
		# input  shape of x is [None,self.N_his*self.dim]
		# output shape of z is [None,           self.dim]
		Disassembly_result = []
		for i in range(self.J):
			temp = x
			for k in range(self.n_d+1):
				temp = self.Disassembly_block[i][k](temp)
			Disassembly_result += [tf.keras.layers.Reshape((-1, self.n_nodes))(temp)]
		x = tf.keras.layers.Concatenate(axis=1)(Disassembly_result)
		Assembly_result = []
		for i in range(self.n_nodes):
			temp = tf.keras.layers.Lambda(lambda x: x[:,:,i])(x)
			for k in range(self.n_a+1):
				temp = self.Assembly_layer[k](temp)
			Assembly_result += [temp]
		z = tf.keras.layers.Concatenate(axis=1)(Assembly_result)
		z = self.out_layer(z)
		return z

	def train(self, inputs_train, outputs_train, temp_model_path, model_path, hist_path):
		# build model
		self.build(input_shape=(None,self.N_his*self.dim))
		# complie model
		if self.lr['type_']=='value':
			self.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.lr['lrate']), loss='mean_squared_error')
			callbacks2 = []
		elif self.lr['type_']=='cyclic':
			lra = self.lr
			self.compile(optimizer='adam', loss='mean_squared_error')
			callbacks2 = [CLr.CyclicLR(base_lr=lra['base'],max_lr=lra['max'],step_size=lra['stepsize'],mode=lra['mode'],gamma=lra['gamma'])]
		else:
			raise AttributeError('ResnetPDEwMJCP2022::train: No this type of learning rate')
		callback = MCp.ModelCheckpoint2(temp_model_path, inputs=inputs_train,outputs=outputs_train,nepoch=self.n_epochs,
										monitor='loss', verbose=0, mode='min', period=self.batch_period)
		callbacks2 += [callback]
		self.summary()
		hist = self.fit(inputs_train, outputs_train, epochs=self.n_epochs,
	              batch_size=self.batch_size, callbacks=callbacks2)
		self.save_weights(model_path)
		json.dump(hist.history, open(hist_path, 'w'), indent=2)
		return hist

	def CompileOnly(self):
		# build model
		self.build(input_shape=(None,self.N_his*self.dim))
		# complie model
		if self.lr['type_']=='value':
			self.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.lr['lrate']), loss='mean_squared_error')
		elif self.lr['type_']=='cyclic':
			lra = self.lr
			self.compile(optimizer='adam', loss='mean_squared_error')
		else:
			raise AttributeError('ResnetPDEwMJCP2022::CompileOnly: No this type of learning rate')

class DataTran():
	def __init__(self,config):
		self.eqn_config = config.eqn_config
		self.net_config = config.net_config
		self.dat_config = config.dat_config
		self.d_RNN  = self.net_config.N_rec
		self.N_his  = self.net_config.N_his
		self.n_ea_traj       = self.dat_config.n_ea_traj
		self.train_data_path = self.dat_config.TrainData_dir
		self.test_data_path  = self.dat_config.TestData_dir

	def read_traindata(self):
		# train data is assumed to be stored under key 'data' of matfile
		# train data in this function is in the form of [dim,n_of_time_step,n_of_tracjectory]
		try:
			self.train_data = (sio.loadmat(self.train_data_path))['data']
		except:
			raise AttributeError('DataTran::read_traindata: Please check data file.')
		self.dim, self.L_Nmax, self.N_long_traj = (self.train_data).shape
		self.n_train = self.n_ea_traj * self.N_long_traj

	def read_testdata(self):
		try:
			self.test_data = (sio.loadmat(self.test_data_path))['data']
		except:
			raise AttributeError('DataTran::read_traindata: Please check data file.')
		self.dim = (self.test_data).shape[0]

	def test_1dat1model(self,model,save_path):
		# only used when single data is tested
		L_Nmax_Test = (self.test_data).shape[-1]
		data_ = (self.test_data).flatten()
		self.pred = self.test_singledata(data_,model,L_Nmax_Test)
		sio.savemat(save_path,{'pred':self.pred})

	def test_1datEnsemAver(self,modellist,save_path):
		# This is a Ensemble averge version of function 'test_1dat1model'
		# only used when single data is tested
		L_Nmax_Test = (self.test_data).shape[-1]
		data_ = (self.test_data).flatten()
		self.pred = self.test_singledataEnsemAver(data_,modellist,L_Nmax_Test)
		sio.savemat(save_path,{'pred':self.pred})

	def test_pertubedat(self,model,save_path):
		# only used when multiple data is tested
		if ('pertube_test' in self.dat_config.keys()) and self.dat_config['pertube_test']['if']:
			logging.info('Pertube data test is conducted in this test')
			try:
				self.pertubetest_data = (sio.loadmat(self.dat_config['pertube_test']['data_dir']))['data']
			except:
				raise AttributeError('DataTran::test_pertubedat: Please check data file.')
			N_pertube,L_Nmax_Test = (self.pertubetest_data).shape[-1],(self.pertubetest_data).shape[1]
			self.pred_all = np.zeros(self.pertubetest_data.shape)
			for i in range(N_pertube):
				data_ = (self.pertubetest_data[:,:,i]).flatten()
				pred  = self.test_singledata(data_,model,L_Nmax_Test)
				self.pred_all[:,:,i] = pred
			sio.savemat(save_path,{'pertube_pred':self.pred_all})
		else:
			pass

	def traindata_rescomp_dim1(self,model,save_path):
		# trandata_pre = np.zeros(self.train_data.shape)
		# for i in range(self.N_long_traj):
		# 	trandata_pre[0,:,i] = self.test_singledata(self.train_data[0,:,i],model,self.L_Nmax)
		trandata_pre = ((self.test_tensordata(self.train_data[0].T,model,self.L_Nmax)).T).reshape(self.train_data.shape)
		self.trandata_res = self.train_data-trandata_pre
		sio.savemat(save_path,{'data':self.trandata_res})

	def train_data_trans(self,seed_):
		smaple_L_Nmax = self.L_Nmax-self.d_RNN-self.N_his
		# random setting
		np.random.seed(seed_)
		sample_init_L = np.random.randint(smaple_L_Nmax+1,size=(self.N_long_traj,self.n_ea_traj))
		temp_wu = np.random.permutation(self.n_train)
		# data merging
		data_ = (np.vstack(self.train_data)).T
		# set train inputs and outputs
		inputs_train = np.zeros((self.n_train, self.dim*self.N_his))
		output_train = np.zeros((self.n_train, self.dim*self.d_RNN))
		for i in range(self.n_ea_traj):
			inputs_train[i*self.N_long_traj:(i+1)*self.N_long_traj] = self.datachoose(data_, self.dim, sample_init_L[:,[i]]           , self.N_his)
			output_train[i*self.N_long_traj:(i+1)*self.N_long_traj] = self.datachoose(data_, self.dim, sample_init_L[:,[i]]+self.N_his, self.d_RNN)
		self.inputs_train  = inputs_train[temp_wu,:]
		self.outputs_train = output_train[temp_wu,:]

	def test_singledata(self,test_data,model,N_T):
		# data is in the form of [dim*n_of_time_step]
		# aranging as [dim1_tracj, dim2_tracj,...]
		data_ = self.datachoose(test_data, self.dim, 0, self.N_his)
		pre = np.zeros(N_T*self.dim)
		pre[:(self.N_his*self.dim)] = data_
		for i in range(N_T-self.N_his):
			next_time = model.predict(np.array([pre[self.dim*i:self.dim*(i+self.N_his)]]))
			pre[self.dim*(i+self.N_his):self.dim*(i+self.N_his+1)] = next_time[:,:self.dim][0]
		pre = (pre.reshape([N_T,self.dim])).T
		return pre

	def test_tensordata(self,test_data,model,N_T):
		# data is in the form of [None,dim*n_of_time_step]
		# aranging as [dim1_tracj, dim2_tracj,...]
		data_ = self.datachoose(test_data, self.dim, 0, self.N_his)
		pre = np.zeros([test_data.shape[0],N_T*self.dim])
		pre[:,:(self.N_his*self.dim)] = data_
		for i in range(N_T-self.N_his):
			next_time = model.predict(pre[:,self.dim*i:self.dim*(i+self.N_his)])
			pre[:,self.dim*(i+self.N_his):self.dim*(i+self.N_his+1)] = next_time[:,:self.dim]
		return pre

	def test_singledataEnsemAver(self,test_data,modellist,N_T):
		# This is a Ensemble averge version of function 'test_singledata'
		# data is in the form of [dim*n_of_time_step]
		# aranging as [dim1_tracj, dim2_tracj,...]
		N_model = len(modellist)
		data_ = self.datachoose(test_data, self.dim, 0, self.N_his)
		pre = np.zeros(N_T*self.dim)
		pre[:(self.N_his*self.dim)] = data_
		for i in range(N_T-self.N_his):
			next_time = 0
			for j in range(N_model):
				next_time += modellist[j].predict(np.array([pre[self.dim*i:self.dim*(i+self.N_his)]]))
			pre[self.dim*(i+self.N_his):self.dim*(i+self.N_his+1)] = (next_time/N_model)[:,:self.dim][0]
		pre = (pre.reshape([N_T,self.dim])).T
		return pre

	def datachoose(self,data,dim,start,nstep):
		# data (1D or 2D) is in the form of [N,dim*n_of_time_step]
		# aranging as N*[dim1_tracj, dim2_tracj,...]
		# this function will choose consecutive 'nstep' of data from 'start' ([N,1]) index
		Ndata = 1 if (data.ndim==1) else data.shape[0]
		traclen = int(data.shape[-1]/dim)
		if (traclen-nstep)<np.max(start):
			raise AttributeError('DataTran::datachoose: Cant take so long step')
		ind = np.tile(np.arange(dim)*traclen,(nstep*Ndata,1))
		temp = (np.tile(np.arange(nstep),Ndata))[:,None]
		ind = (ind+temp).reshape([Ndata,dim*nstep])+start
		if data.ndim==1:
			ind = ind[0]
		return np.take_along_axis(data, ind, axis=(data.ndim-1))

