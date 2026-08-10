from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import munch
import pdb
import logging
import os
import time

import numpy as np
import scipy
import scipy.io as sio
from scipy import stats
import matplotlib
from matplotlib import pyplot as plt
import sklearn
from sklearn import decomposition
import tensorflow as tf
import tqdm

import Evaulation
import GANSde
import ResnetPDEwM

class MixWGANGPSde(tf.keras.Model):
	def __init__(self,config,summary=True):
		super().__init__()
		## parameter set
		self.eqn_config = config.eqn_config
		self.net_config = config.net_config
		self.dim = self.eqn_config.dim
		self.d_RNN        = self.net_config.N_rec
		self.n_Z          = self.net_config.n_Z
		self.G_type       = self.net_config.G_type
		self.D_type       = self.net_config.D_type
		self.G_hidden     = self.net_config.G_hidden
		self.D_hidden     = self.net_config.D_hidden
		self.G_nodes      = self.net_config.G_nodes
		self.D_nodes      = self.net_config.D_nodes
		self.n_epochs     = self.net_config.N_epochs
		self.batch_size   = self.net_config.batch_size
		self.n_critic     = self.net_config.n_critic
		self.G_opt        = self.net_config.G_opt
		self.D_opt        = self.net_config.D_opt
		self.Test_mode    = self.net_config.Test_mode
		## parameter for WGAN-GP
		self.GP_weight    = 10.0
		## build model
		self.build_model(summary)

	def build_model(self,summary=True):
		## Generator
		if self.G_type=='MLP':
			self.G  = self.MLP_G()
		elif self.G_type=='MLP_relu':
			self.G  = self.MLP_G_reluactv()
		else:
			raise AttributeError('build_model: No %s type of Generator'%(G_type))
		## Discriminator
		if self.D_type=='MLP':
			self.D  = self.MLP_D()
		elif self.D_type=='DCGAN':
			self.D  = self.DCGAN_D()
		else:
			raise AttributeError('build_model: No %s type of Discriminator'%(G_type))
		## Model Summary
		if summary:
			self.G.summary()
			self.D.summary()
		## optimization
		self.G_optimizer = self.optimizer_choose(self.G_opt)
		self.D_optimizer = self.optimizer_choose(self.D_opt)
		## test model
		self.test_model = self.test_model_choose(self.Test_mode)

	def train(self, train_data, train_hidden_data, model_path, hist_path, Monitor, DatVes, predt_path):
		logtim = max(1, int(self.n_epochs/10))
		## Saver
		G_loss, D_loss, W_dist = [],[],[]
		checkpoint = tf.train.Checkpoint(G_optimizer=self.G_optimizer,D_optimizer=self.D_optimizer,G=self.G,D=self.D)
		ckptmanager = tf.train.CheckpointManager(checkpoint, model_path, max_to_keep=1)
		if ('Ens_monitor' in Monitor.monitor_config.keys()) and (Monitor.monitor_config.Ens_monitor['if']):
			checkpoint_Ens = tf.train.Checkpoint(G_optimizer=self.G_optimizer,D_optimizer=self.D_optimizer,G=self.G,D=self.D)
			ckptmanager_Ens = tf.train.CheckpointManager(checkpoint_Ens, Monitor.Ens_save_path, max_to_keep=10)
		## train data
		train_data = tf.cast(train_data,tf.float32)
		train_dataset = tf.data.Dataset.from_tensor_slices(train_data).batch(self.batch_size)
		N_batch = int(train_data.shape[0]/self.batch_size)
		## train
		for epoch in range(self.n_epochs):
			for batch,train_x in tqdm.tqdm(zip(np.arange(N_batch)+1,train_dataset), total=N_batch):
				self.train_step(train_x,batch)
			D_lossdata, G_lossdata, W_distdata = self.compute_loss(train_data)
			# negative distance
			W_distdata = -W_distdata
			G_loss.append(G_lossdata.numpy())
			D_loss.append(D_lossdata.numpy())
			W_dist.append(W_distdata.numpy())
			print("Epoch: {} | disc_loss: {} | gen_loss: {} | W_dist: {}".format(epoch, D_lossdata, G_lossdata, W_distdata))
			# save model
			if (epoch + 1) % 2000 == 0:
				ckptmanager.save()
			if (epoch + 1) % logtim == 0:
				logging.info('Epoch %d/%d has been reached'%(epoch+1,self.n_epochs))
			# monitor
			if Monitor.monitor_config.pdf_monitor['if']:
				Monitor.condpdf_plotting(self,epoch)
			if Monitor.monitor_config.repdf_display['if']:
				Monitor.complete_condpdf(self,epoch)
			if Monitor.monitor_config.fake_check['if']:
				Monitor.fakesample_check(self,train_data,epoch)
			if Monitor.monitor_config.cond_mv['if']:
				Monitor.cond_meanvar(self,epoch)
			if Monitor.monitor_config.loss['if']:
				Monitor.Eva_loss(self,epoch,G_loss,D_loss)
			if Monitor.monitor_config.Evameanv['if']:
				if Monitor.monitor_config.Evameanv['type']=="Normal":
					Monitor.Eva_meanv(self,epoch,DatVes,predt_path)
				elif Monitor.monitor_config.Evameanv['type']=="Multiple_last":
					Monitor.Eva_meanv_Multiple_last(self,epoch,DatVes,predt_path)
			if Monitor.monitor_config.Ens_monitor['if']:
				Monitor.Ens_monitor(epoch,ckptmanager_Ens,checkpoint_Ens,self,DatVes)
			# test model
			self.test_model(DatVes, predt_path, epoch)
		## save results
		ckptmanager.save()
		json.dump({'G_loss':(np.float_(G_loss)).tolist(), 'D_loss': (np.float_(D_loss)).tolist(), 'W_dist': (np.float_(W_dist)).tolist()}, open(hist_path, 'w'),indent=2)

	def MLP_G(self):
		G_layer  = [tf.keras.layers.InputLayer(input_shape=(self.dim+self.n_Z))]
		G_layer += [tf.keras.layers.Dense(self.G_nodes, activation='tanh') for __ in range(self.G_hidden)]
		G_layer += [tf.keras.layers.Dense(self.dim)]
		return tf.keras.Sequential(G_layer,name='Generator')

	def MLP_G_reluactv(self):
		G_layer  = [tf.keras.layers.InputLayer(input_shape=(self.dim+self.n_Z))]
		G_layer += [tf.keras.layers.Dense(self.G_nodes, activation='relu') for __ in range(self.G_hidden)]
		G_layer += [tf.keras.layers.Dense(self.dim)]
		return tf.keras.Sequential(G_layer,name='Generator')

	def MLP_D(self):
		D_layer  = [tf.keras.layers.InputLayer(input_shape=(self.d_RNN*self.dim))]
		D_layer += [tf.keras.layers.Dense(self.D_nodes, activation='tanh') for __ in range(self.D_hidden)]
		D_layer += [tf.keras.layers.Dense(1)]
		return tf.keras.Sequential(D_layer,name='Discriminator')

	def DCGAN_D(self):
		input_x_layer = tf.keras.layers.Input((self.d_RNN*self.dim))
		x = tf.keras.layers.Reshape((self.d_RNN*self.dim,1,1))(input_x_layer)
		x = tf.keras.layers.Conv2D(64, (4, 4), strides=(2, 2), padding='same', use_bias=False, kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02))(x)
		#x = LayerNormalization()(x)
		x = tf.keras.layers.LeakyReLU()(x)
		x = tf.keras.layers.Conv2D(128, (4, 4), strides=(2, 2), padding='same', use_bias=False, kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02))(x)
		#x = LayerNormalization()(x)
		x = tf.keras.layers.LeakyReLU()(x)
		# x = tf.keras.layers.Conv2D(256, (4, 4), strides=(2, 2), padding='same', use_bias=False, kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02))(x)
		# #x = LayerNormalization()(x)
		# x = tf.keras.layers.LeakyReLU()(x)
		# x = tf.keras.layers.Conv2D(512, (4, 4), strides=(2, 2), padding='same', use_bias=False, kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02))(x)
		# #x = LayerNormalization()(x)
		# x = tf.keras.layers.LeakyReLU()(x)
		x = tf.keras.layers.Conv2D(1, (4, 4), strides=(1, 1), padding='same', use_bias=False, kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02))(x)
		x = tf.keras.layers.Flatten()(x)
		output = tf.keras.layers.Dense(1)(x)
		model = tf.keras.Model(inputs=input_x_layer, outputs=output)
		return model

	def Generate(self,x,Z):
		# Input shape:  x: [None, self.dim]
		#               Z: [None, self.n_Z]
		# Output shape:    [None, self.dim]
		return self.G(tf.concat([x, Z], -1))+self.Model_drift.myevaluateincrem(x)

	def Discriminate(self,x):
		# Input shape:  x: [None, self.d_RNN*self.dim]
		# Output shape:    [None, 1]
		return self.D(x)

	def compute_loss(self, x):
		# generating samples
		Xs = x[:,:self.dim]  ##### check dimension
		X_sample = [Xs] 
		for i in range(self.d_RNN-1):
			Z_samp = tf.random.normal([x.shape[0], self.n_Z])
			Xinc = self.Generate(Xs,Z_samp) ##### check dimension
			X_sample += [Xinc]
			Xs = Xs+Xinc
		X_sample = tf.concat(X_sample, -1)
		# discriminate x and x_gen
		logits_true = self.Discriminate(x)
		logits_fake = self.Discriminate(X_sample)
		# gradient penalty
		d_regularizer = self.gradient_penalty(x, X_sample)
		# losses
		W_dist = tf.reduce_mean(logits_fake)-tf.reduce_mean(logits_true)
		D_loss = W_dist+d_regularizer*self.GP_weight
		G_loss  = -tf.reduce_mean(logits_fake)
		return D_loss, G_loss, W_dist

	def gradient_penalty(self, x, x_gen):
		epsilon = tf.random.uniform([x.shape[0], 1], 0.0, 1.0)
		x_hat = epsilon*x+(1-epsilon)*x_gen
		with tf.GradientTape() as t:
			t.watch(x_hat)
			d_hat = self.Discriminate(x_hat)
		gradients = t.gradient(d_hat, x_hat)
		ddx = tf.sqrt(tf.reduce_sum(gradients**2, axis=1)+1.0e-12)
		d_regularizer = tf.reduce_mean((ddx - 1.0)**2)
		return d_regularizer

	def predict(self,x):
		Z = tf.random.normal([x.shape[0], self.n_Z])
		return x+self.Model_drift.myevaluateincrem(x)+self.G(tf.concat([x, Z], -1))

	def optimizer_choose(self,opti):
		if opti['type_']=='Adam':
			return tf.keras.optimizers.legacy.Adam(opti['lr'],beta_1=opti['beta_1'],beta_2=opti['beta_2'])
		elif opti['type_']=='RMSprop':
			return tf.keras.optimizers.legacy.RMSprop(opti['lr'])
		else:
			raise AttributeError('optimizer_choose: Now do not support this optimizer')

	def test_model_choose(self, Test_mode):
		if Test_mode=='Normal':
			return self.Test_last
		elif Test_mode=='Multiple_last':
			self.Testepoches = self.Last_epochs(20,10,self.n_epochs)
			return self.Test_Multiple_last
		else:
			raise AttributeError('test_model: Do not support %s type Test'%(Test_mode))

	def Test_last(self, DatVes, predt_path, epoch):
		if epoch+1==self.n_epochs:
			logging.info('Test on Epoch %d'%(epoch+1))
			DatVes.test_mdat1model(self,predt_path)

	def Test_Multiple_last(self, DatVes, predt_path, epoch):
		if epoch+1 in self.Testepoches:
			if (epoch+1)==self.Testepoches[0]:
				logging.info('Ensemble Test on Epoch %d'%(epoch+1))
				DatVes.test_mdat1model(self,predt_path,mode='w')
			else:
				logging.info('Ensemble Test on Epoch %d'%(epoch+1))
				DatVes.test_mdat1model(self,predt_path,mode='a')

	def Last_epochs(self,n1,n2,nepoch):
		# for nepoch epoches, return the ** # of epoch ** before last one (including last)
		step = int(nepoch/(n1*n2))
		# if nepoch too small, then degenerate to Test_last
		if step==0:
			return np.array((nepoch))
		initial = nepoch-n2*step
		return (np.arange(n2)+1)*step+initial

	@tf.function
	def train_step(self, x, iter_):
		with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
			disc_loss, gen_loss, __ = self.compute_loss(x)
		# update D gradients
		disc_gradients = disc_tape.gradient(disc_loss, self.D.trainable_variables)
		self.D_optimizer.apply_gradients(zip(disc_gradients, self.D.trainable_variables))

		# update G gradients
		if iter_%self.n_critic==0:
			gen_gradients = gen_tape.gradient(gen_loss, self.G.trainable_variables)
			self.G_optimizer.apply_gradients(zip(gen_gradients, self.G.trainable_variables))

class Monitor(GANSde.Monitor):
	def fake_sample(self,model,data):
		Xs = data[:,:model.dim]
		X_sample = [Xs] 
		for i in range(model.d_RNN-1):
			Z_samp = tf.random.normal([data.shape[0], model.n_Z])
			Xinc = model.Generate(Xs,Z_samp)
			X_sample += [Xinc]
			Xs = Xs + Xinc
		X_sample = np.concatenate(X_sample, -1)
		return X_sample

	def readMultiplemodel(self,path,ckptmanager,model):
		# This function is designed for test for multiple models
		modellist = []
		modeldict = ckptmanager.checkpoints
		for i in range(len(modeldict)):
			ModelX = self.GANsModel(self.config,summary=False)
			ModelX.Model_drift = model.Model_drift
			# see https://www.tensorflow.org/api_docs/python/tf/train/Checkpoint
			ModelXCheckp = tf.train.Checkpoint(G_optimizer=ModelX.G_optimizer,D_optimizer=ModelX.D_optimizer,G=ModelX.G,D=ModelX.D)
			# manager = tf.train.CheckpointManager(ModelXCheckp, path, max_to_keep=1)
			ModelXCheckp.restore(modeldict[i]).expect_partial()
			modellist.append(ModelX)
		return modellist

	def Mulmodel_Generate(self,modellist,Xs):
		Nmodel = len(modellist)
		modelid = np.random.randint(Nmodel, size=Xs.shape[0])
		Xre = np.zeros(Xs.shape)
		for j in range(Nmodel):
			_id = np.where(modelid==j)[0]
			Z_samp = tf.random.normal([_id.shape[0], modellist[0].n_Z])
			Xre[_id] = Xs[_id]+modellist[j].Generate(Xs[_id],Z_samp)
		return Xre

class DataTran(GANSde.DataTran):
	def __init__(self,config,Monitor=None):
		# Note: d_RNN here denotes the time-length of data (diff with ResNetPDE)
		self.eqn_config = config.eqn_config
		self.net_config = config.net_config
		self.dat_config = config.dat_config
		self.d_RNN  = self.net_config.N_rec
		self.n_Z    = self.net_config.n_Z
		self.n_ea_traj       = self.dat_config.n_ea_traj
		self.train_data_path = self.dat_config.TrainData_dir
		self.test_data_path  = self.dat_config.TestData_dir
		self.resmodel_path   = self.eqn_config.resmodel_path
		self.resconfig_path  = self.eqn_config.resconfig_path
		self.N_pred          = self.dat_config.N_pred
		self.Monitor         = Monitor

	def test_mdat1model(self,model,save_path,mode='w'):
		L_Nmax_Test = (self.test_data).shape[1]
		self.pred = self.test_tensordata(self.test_data,model,L_Nmax_Test)
		if mode=='w':
			sio.savemat(save_path,{'pred':self.pred})
		elif mode=='a':
			if os.path.exists(save_path):
				data_exist = (sio.loadmat(save_path))['pred']
				self.pred = np.concatenate([data_exist,self.pred],axis=-1)
			sio.savemat(save_path,{'pred':self.pred})

	def train_hiddendata(self):
		if self.eqn_config.resmodel=='ResNetwM':
			ResModel = ResnetPDEwM.ResnetPDEwM
			with open(self.resconfig_path) as json_data_file:
				resmodel_config = json.load(json_data_file)
			resmodel_config = munch.munchify(resmodel_config)
			self.Model_drift = self.readmodelKeras(self.resmodel_path,ResModel,resmodel_config)
		elif self.eqn_config.resmodel=='Exact':
			if self.eqn_config.eqn_name=='Trig_drift':
				self.Model_drift = Trig_drift_exact_drift(self.eqn_config.k,self.eqn_config.sigma,self.eqn_config.Delta)
			elif self.eqn_config.eqn_name in ['MdOU','SO']:
				self.Model_drift = MdOU_exact_drift(self.eqn_config.mu,self.eqn_config.Delta)
			elif self.eqn_config.eqn_name=='SPDE_Ex1SHeatEqu':
				self.Model_drift = SHeatEqu_exact_drift(self.eqn_config.dim,self.eqn_config.epsilon,self.eqn_config.h,self.eqn_config.Delta)
			elif self.eqn_config.eqn_name=='Geometric Brownian Motion':
				self.Model_drift = GBM_drift_exact_drift(self.eqn_config.mu,self.eqn_config.Delta)
		else:
			raise AttributeError('train_hiddendata::No this model')
		self.train_hiddendata = 0
		# change learned data to increment
		self.train_mat[:,self.dim:] = self.train_mat[:,self.dim:]-self.train_mat[:,:-self.dim]

	def test_tensordata(self,test_data,model,N_T):
		# data is in the form of [dim*n_of_time_step]
		# aranging as [dim1_tracj, dim2_tracj,...]
		data_ = self.datachoose((np.vstack(test_data)).T, self.dim, np.zeros([test_data.shape[-1],1],dtype=int), 1)
		Xs = data_[:,:self.dim]
		pre = [Xs] 
		for i in range(N_T-1):
			Z_samp = tf.random.normal([data_.shape[0], model.n_Z])
			Xs = Xs+model.Generate(Xs,Z_samp)
			pre += [Xs]
		pre = np.concatenate(pre, -1)
		pre_ = np.zeros([self.dim,N_T,self.N_pred])
		for j in range(self.dim):
			pre_[j] = (pre[:,j::self.dim]).T
		return pre_

	def readmodelKeras(self,path,Model,config):
		# This function is designed for test for single models
		ModelX = Model(config)
		# see https://www.tensorflow.org/tutorials/keras/save_and_load?hl=zh-cn
		ModelX.build_compile()
		ModelX.load_weights(path).expect_partial()
		return ModelX

class Trig_drift_exact_drift:
	def __init__(self,k,sigma,Delta):
		self.k = tf.constant(k,dtype='float32')
		self.sigma = tf.constant(sigma,dtype='float32')
		self.Delta = tf.constant(Delta,dtype='float32')
	def myevaluate(self,x):
		return x+self.Delta*tf.cast(tf.math.sin(2*np.pi*self.k*x),'float32')
	def myevaluateincrem(self,x):
		return self.Delta*tf.cast(tf.math.sin(2*np.pi*self.k*x),'float32')

class MdOU_exact_drift:
	def __init__(self,mu,Delta):
		self.mu = tf.constant(mu,dtype='float32')
		self.muT = tf.constant(np.array(mu).T,dtype='float32')
		self.Delta = tf.constant(Delta,dtype='float32')
	def myevaluate(self,x):
		return x+self.Delta*tf.cast(x@self.muT,'float32')
	def myevaluateincrem(self,x):
		return self.Delta*tf.cast(x@self.muT,'float32')

class SHeatEqu_exact_drift:
	def __init__(self,dim,epsilon,h,Delta):
		self.epsilon = tf.constant(epsilon,dtype='float32')
		A = (np.diag(np.full(dim,2))-np.diag(np.ones(dim-1),1)-np.diag(np.ones(dim-1),-1))/h**2
		M = np.linalg.inv(np.eye(dim)+Delta*epsilon*A)
		self.MT = tf.constant(M.T,dtype='float32')
		self.Delta = tf.constant(Delta,dtype='float32')
	def myevaluate(self,x):
		return tf.cast(x@self.MT,'float32')
	def myevaluateincrem(self,x):
		return tf.cast(x@self.MT,'float32')-x

class GBM_drift_exact_drift:
	def __init__(self,mu,Delta):
		self.mu = tf.constant(Delta,dtype='float32')
		self.Delta = tf.constant(Delta,dtype='float32')
	def myevaluate(self,x):
		return x+self.Delta*tf.cast(self.mu*x,'float32')
	def myevaluateincrem(self,x):
		return self.Delta*tf.cast(self.mu*x,'float32')
