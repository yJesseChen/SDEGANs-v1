from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import pdb
import logging
import os

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

try:
	import seaborn as sns
except:
	pass

import Evaulation

class GANSde(tf.keras.Model):
	def __init__(self,config):
		super().__init__()
		## parameter set
		self.eqn_config = config.eqn_config
		self.net_config = config.net_config
		self.dim = self.eqn_config.dim
		self.d_RNN        = self.net_config.N_rec
		self.n_Z          = self.net_config.n_Z
		self.G_hidden     = self.net_config.G_hidden
		self.D_hidden     = self.net_config.D_hidden
		self.G_nodes      = self.net_config.G_nodes
		self.D_nodes      = self.net_config.D_nodes
		self.n_epochs     = self.net_config.N_epochs
		self.batch_size   = self.net_config.batch_size
		## build model
		self.build_model()

	def build_model(self):
		## Generator
		G_layer  = [tf.keras.layers.InputLayer(input_shape=(self.dim+self.n_Z))]
		G_layer += [tf.keras.layers.Dense(self.G_nodes, activation='tanh') for __ in range(self.G_hidden)]
		G_layer += [tf.keras.layers.Dense(self.dim)]
		self.G   = tf.keras.Sequential(G_layer,name='Generator')
		## Discriminator
		D_layer  = [tf.keras.layers.InputLayer(input_shape=(self.d_RNN*self.dim))]
		D_layer += [tf.keras.layers.Dense(self.D_nodes, activation='relu') for __ in range(self.D_hidden)]
		D_layer += [tf.keras.layers.Dense(1)]
		self.D   = tf.keras.Sequential(D_layer,name='Discriminator')
		## Model Summary
		self.G.summary()
		self.D.summary()
		## optimization
		self.G_optimizer = tf.keras.optimizers.Adam(0.002)
		self.D_optimizer = tf.keras.optimizers.RMSprop(0.001)

	def train(self, train_data, model_path, hist_path):
		logtim = int(self.n_epochs/10)
		## Saver
		G_loss, D_loss = [],[]
		checkpoint = tf.train.Checkpoint(G_optimizer=self.G_optimizer,D_optimizer=self.D_optimizer,G=self.G,D=self.D)
		ckptmanager = tf.train.CheckpointManager(checkpoint, model_path, max_to_keep=1)
		## train data
		train_data = tf.cast(train_data,tf.float32)
		train_dataset = tf.data.Dataset.from_tensor_slices(train_data).batch(self.batch_size)
		N_batch = int(train_data.shape[0]/self.batch_size)
		## train
		for epoch in range(self.n_epochs):
			for train_x in tqdm.tqdm(train_dataset, total=N_batch):
				self.train_step(train_x)
			D_lossdata, G_lossdata = self.compute_loss(train_data)
			G_loss.append(G_lossdata.numpy())
			D_loss.append(D_lossdata.numpy())
			print("Epoch: {} | disc_loss: {} | gen_loss: {}".format(epoch, D_lossdata, G_lossdata))
			# save model
			if (epoch + 1) % 300 == 0:
				ckptmanager.save()
			if (epoch + 1) % logtim == 0:
				logging.info('Epoch %d/%d has been reached'%(epoch+1,self.n_epochs))
		## save results
		ckptmanager.save()
		json.dump({'G_loss':(np.float_(G_loss)).tolist(), 'D_loss': (np.float_(D_loss)).tolist()}, open(hist_path, 'w'),indent=2)

	def Generate(self,x,Z):
		# Input shape:  x: [None, self.dim]
		#               Z: [None, self.n_Z]
		# Output shape:    [None, self.dim]
		# pdb.set_trace()
		return self.G(tf.concat([x, Z], -1))

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
			Xs = self.Generate(Xs,Z_samp) ##### check dimension
			X_sample += [Xs]
		X_sample = tf.concat(X_sample, -1)
		# discriminate x and x_gen
		logits_true = self.Discriminate(x)
		logits_fake = self.Discriminate(X_sample)
		# losses computing
		# losses of real with label "1"
		D_real_loss = self.gan_loss(logits=logits_true, is_real=True)
		# losses of fake with label "0"
		D_fake_loss = self.gan_loss(logits=logits_fake, is_real=False)
		D_loss = D_fake_loss + D_real_loss
		# losses of fake with label "1"
		G_loss = self.gan_loss(logits=logits_fake, is_real=True)
		return D_loss, G_loss

	def compute_gradients(self, x):
		with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
			disc_loss, gen_loss = self.compute_loss(x)
		# compute gradients
		gen_gradients = gen_tape.gradient(gen_loss, self.G.trainable_variables)
		disc_gradients = disc_tape.gradient(disc_loss, self.D.trainable_variables)
		return gen_gradients, disc_gradients

	def apply_gradients(self, gen_gradients, disc_gradients):
		self.G_optimizer.apply_gradients(zip(gen_gradients, self.G.trainable_variables))
		self.D_optimizer.apply_gradients(zip(disc_gradients, self.D.trainable_variables))

	def gan_loss(self, logits, is_real=True):
		# Computes standard gan loss between logits and labels
		if is_real:
			labels = tf.ones_like(logits)
		else:
			labels = tf.zeros_like(logits)
		return tf.compat.v1.losses.sigmoid_cross_entropy(multi_class_labels=labels, logits=logits)

	def predict(self,x):
		Z = tf.random.normal([x.shape[0], self.n_Z])
		return self.Generate(x,Z)

	def optimizer_choose(self,opti):
		if opti['type_']=='Adam':
			return tf.keras.optimizers.Adam(opti['lr'],beta_1=opti['beta_1'],beta_2=opti['beta_2'])
		elif opti['type_']=='RMSprop':
			return tf.keras.optimizers.RMSprop(opti['lr'])
		else:
			raise AttributeError('optimizer_choose: Now do not support this optimizer')

	@tf.function
	def train_step(self, train_x):
		gen_gradients, disc_gradients = self.compute_gradients(train_x)
		self.apply_gradients(gen_gradients, disc_gradients)

class WGANGPSde(tf.keras.Model):
	def __init__(self,config):
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
		self.build_model()

	def build_model(self):
		## Generator
		if self.G_type=='MLP':
			self.G  = self.MLP_G()
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
		self.G.summary()
		self.D.summary()
		## optimization
		self.G_optimizer = self.optimizer_choose(self.G_opt)
		self.D_optimizer = self.optimizer_choose(self.D_opt)
		## test model
		self.test_model = self.test_model_choose(self.Test_mode)

	def train(self, train_data, model_path, hist_path, Monitor, DatVes, predt_path):
		logtim = int(self.n_epochs/10)
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
		# pdb.set_trace()
		return self.G(tf.concat([x, Z], -1))

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
			Xs = self.Generate(Xs,Z_samp) ##### check dimension
			X_sample += [Xs]
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

	# def compute_gradients(self, x):
	# 	with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
	# 		disc_loss, gen_loss, __ = self.compute_loss(x)
	# 	# compute gradients
	# 	gen_gradients = gen_tape.gradient(gen_loss, self.G.trainable_variables)
	# 	disc_gradients = disc_tape.gradient(disc_loss, self.D.trainable_variables)
	# 	return gen_gradients, disc_gradients

	# def apply_gradients(self, gen_gradients, disc_gradients):
	# 	self.G_optimizer.apply_gradients(zip(gen_gradients, self.G.trainable_variables))
	# 	self.D_optimizer.apply_gradients(zip(disc_gradients, self.D.trainable_variables))

	def predict(self,x):
		Z = tf.random.normal([x.shape[0], self.n_Z])
		return self.Generate(x,Z)

	def optimizer_choose(self,opti):
		if opti['type_']=='Adam':
			return tf.keras.optimizers.Adam(opti['lr'],beta_1=opti['beta_1'],beta_2=opti['beta_2'])
		elif opti['type_']=='RMSprop':
			return tf.keras.optimizers.RMSprop(opti['lr'])
		else:
			raise AttributeError('optimizer_choose: Now do not support this optimizer')

	def test_model_choose(self, Test_mode):
		if Test_mode=='Normal':
			return self.Test_last
		elif Test_mode=='Multiple_last':
			self.Testepoches = self.Last_epochs(10,10,self.n_epochs)
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
		# the step size is nepoch/(n1*n2)
		# n2 is the number of these epochs
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

	# def train_step(self, train_x):
	# 	gen_gradients, disc_gradients = self.compute_gradients(train_x)
	# 	self.apply_gradients(gen_gradients, disc_gradients)

class Monitor():
	def __init__(self,path,config,GANsModel,Evaulation=None):
		self.eqn_config      = config.eqn_config
		self.net_config      = config.net_config
		self.monitor_config  = config.monitor_config
		self.rawpath         = path
		self.pdfpath         = self.rawpath+'pdfplot'
		self.repdfpath       = self.rawpath+'repdfplot'
		self.dataplotpath    = self.rawpath+'dataplot'
		self.pcapath         = self.rawpath+'pcaplot'
		self.fakepdfpath     = self.rawpath+'fakepdf'
		self.fakemvpath      = self.rawpath+'fakemv'
		self.cond_mvpath     = self.rawpath+'condmeanvar'
		self.loss_path       = self.rawpath+'loss'
		self.Ens_save_path   = self.rawpath+'Ens_model/'
		self.Ens_cond_mvpath = self.rawpath+'Ens_cond_mv'
		self.Ens_repdfpath   = self.rawpath+'Ens_repdf'
		self.Ens_evapath     = self.rawpath+'Ens_Eva'
		self.Ens_endpdfpath  = self.rawpath+'Ens_Epdf'
		if Evaulation!=None:
			self.Evaulation = Evaulation
		if not os.path.exists(path):
			os.makedirs(path)
		if (self.monitor_config.pdf_monitor['if']) and (not os.path.exists(self.pdfpath)):
			os.makedirs(self.pdfpath)
		if (self.monitor_config.repdf_display['if']) and (not os.path.exists(self.repdfpath)):
			os.makedirs(self.repdfpath)
		if (self.monitor_config.traindata_hist) and (not os.path.exists(self.dataplotpath)):
			os.makedirs(self.dataplotpath)
		if (self.monitor_config.traintransin_hist) and (not os.path.exists(self.dataplotpath)):
			os.makedirs(self.dataplotpath)
		if (self.monitor_config.fake_check['if']) and (not os.path.exists(self.pcapath)):
			os.makedirs(self.pcapath)
		if (self.monitor_config.fake_check['if']) and (not os.path.exists(self.fakepdfpath)):
			os.makedirs(self.fakepdfpath)
		if (self.monitor_config.fake_check['if']) and (not os.path.exists(self.fakemvpath)):
			os.makedirs(self.fakemvpath)
		if (self.monitor_config.cond_mv['if']) and (not os.path.exists(self.cond_mvpath)):
			os.makedirs(self.cond_mvpath)
		if (self.monitor_config.loss['if']) and (not os.path.exists(self.loss_path)):
			os.makedirs(self.loss_path)	
		if (self.monitor_config.Ens_monitor['if']) and (not os.path.exists(self.Ens_save_path)):
			os.makedirs(self.Ens_save_path)
		if (self.monitor_config.Ens_monitor['Ens_cond_mv']) and (not os.path.exists(self.Ens_cond_mvpath)):
			os.makedirs(self.Ens_cond_mvpath)	
		if (self.monitor_config.Ens_monitor['Ens_repdf']) and (not os.path.exists(self.Ens_repdfpath)):
			os.makedirs(self.Ens_repdfpath)	
		if (self.monitor_config.Ens_monitor['Ens_eva']) and (not os.path.exists(self.Ens_evapath)):
			os.makedirs(self.Ens_evapath)
		# if (self.monitor_config.Ens_monitor['Ens_endpdf']) and (not os.path.exists(self.Ens_endpdfpath)):
		# 	os.makedirs(self.Ens_endpdfpath)
		# operate config
		self.condpdf_plotting_points = np.array(self.monitor_config.pdf_monitor['points'])
		self.Delta    = self.eqn_config.Delta
		self.N_epochs = self.net_config.N_epochs
		# global operation
		self.config = config
		self.GANsModel = GANsModel

	def condpdf_plotting(self,model,epoch):
		if (epoch==0) or ((epoch+1)%(self.monitor_config.pdf_monitor['period'])==0):
			N_plot = (self.condpdf_plotting_points).shape[0]
			int_long = self.monitor_config.pdf_monitor['int_long']
			fig, axes = plt.subplots(ncols=N_plot, figsize=(N_plot*3, 2), constrained_layout=True, squeeze=False)
			for n in range(N_plot):
				axes[0,n].set_title("$X_s=$%.2f, ite %d"%(self.condpdf_plotting_points[n],epoch+1))
				self.condpdf_plotting_std(self.eqn_config.eqn_name,axes[0,n],int_long,self.condpdf_plotting_points[n],self.Delta)
				self.condpdf_plotting_data(self.eqn_config.eqn_name,axes[0,n],model,int_long,self.condpdf_plotting_points[n],self.Delta)
			fig.savefig(self.pdfpath+'/'+str(epoch+1)+'.png',dpi=150)
			plt.close()
		else:
			pass

	def complete_condpdf(self,model,epoch,enforce=False):
		if (epoch+1==self.N_epochs) or ((epoch+1)%(int(self.N_epochs/self.monitor_config.repdf_display['times']))==0) or (epoch==0) or enforce:
			# logging.info('--------------Plotting final pdf on Epoch %d'%(epoch+1))
			## check if model list
			path = self.repdfpath if type(model) != list else self.Ens_repdfpath
			if self.eqn_config.dim==1:
				## draw
				int_long = self.monitor_config.repdf_display['int_long']
				p_size = self.monitor_config.repdf_display['size']
				px,py = p_size
				l1,l2 = self.monitor_config.repdf_display['range']
				p_grid = (np.linspace(l1,l2,px*py)).reshape([px,py])
				fig, axes = plt.subplots(nrows=px, ncols=py, figsize=(py*3, px*2), constrained_layout=True, squeeze=False)
				for i in range(px):
					for j in range(py):
						axes[i,j].set_title("$X_s=$%.2f, ite %d"%(p_grid[i,j],epoch+1))
						self.condpdf_plotting_std(self.eqn_config.eqn_name,axes[i,j],int_long,p_grid[i,j],self.Delta)
						self.condpdf_plotting_data(self.eqn_config.eqn_name,axes[i,j],model,int_long,p_grid[i,j],self.Delta)
				fig.savefig(path+'/finalpdf'+str(epoch+1)+'.png',dpi=150)
				plt.close()
				## draw
				# logging.info('--------------End plotting final pdf on Epoch %d'%(epoch+1))
			elif self.eqn_config.dim==2:
				if self.eqn_config.eqn_name in ['MdOU','SO']:
					level = [0,3,6,9,12,15,18,21,24]
				else:
					raise AttributeError('complete_condpdf: no this type of 2D example')
				## draw
				int_long = self.monitor_config.repdf_display['int_long']
				p_size = self.monitor_config.repdf_display['size']
				px,py = p_size
				l1,l2 = self.monitor_config.repdf_display['range']
				p_gridx = np.linspace(l1[0],l1[1],px)
				p_gridy = np.linspace(l2[0],l2[1],py)
				p_gridx,p_gridy = np.meshgrid(p_gridx,p_gridy)
				p_grid = np.array((p_gridx.flatten(),p_gridy.flatten())).T
				fig, axes = plt.subplots(nrows=px, ncols=py*2, figsize=(py*3*2, px*2), constrained_layout=True, squeeze=False)
				for i in range(px):
					for j in range(py):
						axes[i,j*2].set_title("$X_s=$(%.2f,%.2f), ite %d"%(p_grid[(i*py+j)][0],p_grid[(i*py+j)][1],epoch+1))
						axes[i,j*2+1].set_title("$X_s=$(%.2f,%.2f), ite %d"%(p_grid[(i*py+j)][0],p_grid[(i*py+j)][1],epoch+1))
						# self.condpdf_plotting_std2D(self.eqn_config.eqn_name,axes[i,j*2],int_long,p_grid[(i*py+j)],self.Delta,level)
						# self.condpdf_plotting_data2D(self.eqn_config.eqn_name,axes[i,j*2+1],model,int_long,p_grid[(i*py+j)],self.Delta,level)
						self.condmargpdf_plotting_std2D(self.eqn_config.eqn_name,axes[i,j*2],axes[i,j*2+1],int_long,p_grid[(i*py+j)],self.Delta)
						self.condmargpdf_plotting_data2D(self.eqn_config.eqn_name,axes[i,j*2],axes[i,j*2+1],model,int_long,p_grid[(i*py+j)],self.Delta)
				fig.savefig(path+'/finalpdf'+str(epoch+1)+'.png',dpi=150)
				plt.close()
			else:
				pass
		else:
			pass

	def Eva_Ensemble(self,modellist,DatVes,epoch):
		N_T = (DatVes.test_data).shape[1]
		data_ = DatVes.datachoose((np.vstack(DatVes.test_data)).T, DatVes.dim, np.zeros([DatVes.test_data.shape[-1],1],dtype=int), 1)
		Xs = np.tile(data_[:,:DatVes.dim],(len(modellist),1))
		pre = [Xs]
		for i in range(N_T-1):
			Xs = self.Mulmodel_Generate(modellist,Xs)
			pre += [Xs]
		pre = np.concatenate(pre, -1)
		pre_ = np.zeros([DatVes.dim,N_T,DatVes.N_pred*len(modellist)])
		for j in range(self.eqn_config.dim):
			pre_[j] = (pre[:,j::DatVes.dim]).T
		save_ = (self.Ens_evapath+'/'+str(epoch+1)+'M'+'.pdf')
		fig,ax = self.Evaulation.plot_meanstd(DatVes.test_data,pre_,self.eqn_config.dim,self.eqn_config.Delta,savepath=save_)
		# for i in range(min(DatVes.dim,10)):
		# 	save_ = (self.Ens_evapath+'/'+str(epoch+1)+'M'+str(i+1)+'.pdf')
		# 	fig,ax = self.Evaulation.plot_meanstd(DatVes.test_data,pre_,self.eqn_config.dim,self.eqn_config.Delta,savepath=save_)
		# save_ = (self.Ens_evapath+'/'+str(epoch+1)+'M'+'.pdf')
		# fig,ax = self.Evaulation.plot_meanstdGeneralD(DatVes.test_data[:,-1,:],pre_,DatVes.dim,self.eqn_config.Delta,savepath=save_)
		plt.close()

	def Endpdf_Ensemble(self,modellist,DatVes,epoch):
		data_ = DatVes.datachoose((np.vstack(DatVes.test_data)).T, DatVes.dim, np.zeros([DatVes.test_data.shape[-1],1],dtype=int), 1)
		Xs = np.tile(data_[:,:DatVes.dim],(len(modellist),1))
		for i in range(modellist[0].d_RNN-1):
			Xs = self.Mulmodel_Generate(modellist,Xs)
		pre_ = Xs.T
		save_ = (self.Ens_endpdfpath+'/'+str(epoch+1)+'P'+'.pdf')
		fig,ax = self.Evaulation.plot_endpdfGeneralD(DatVes.test_data[:,-1,:],pre_,DatVes.dim,savepath=save_)
		plt.close()

	def Mulmodel_Generate(self,modellist,Xs):
		Nmodel = len(modellist)
		modelid = np.random.randint(Nmodel, size=Xs.shape[0])
		Xre = np.zeros(Xs.shape)
		for j in range(Nmodel):
			_id = np.where(modelid==j)[0]
			Z_samp = tf.random.normal([_id.shape[0], modellist[j].n_Z])
			Xre[_id] = modellist[j].Generate(Xs[_id],Z_samp)
		return Xre

	def cond_meanvar(self,model,epoch,enforce=False):
		if (epoch+1==self.N_epochs) or ((epoch+1)%(int(self.N_epochs/self.monitor_config.cond_mv['times']))==0) or (epoch==0) or enforce:
			## check if model list
			path = self.cond_mvpath if type(model) != list else self.Ens_cond_mvpath
			if self.eqn_config.dim==1:
				## compute
				Npoint = self.monitor_config.cond_mv['Npoint']
				l1,l2 = self.monitor_config.cond_mv['range']
				p_grid = np.linspace(l1,l2,Npoint+1)
				Mean_t, Std_t = np.zeros(p_grid.shape),np.zeros(p_grid.shape)
				Mean_d, Std_d = np.zeros(p_grid.shape),np.zeros(p_grid.shape)
				for i in range(p_grid.shape[0]):
					Mean_t[i],Std_t[i] = self.condmv_plotting_std(self.eqn_config.eqn_name,p_grid[i],self.Delta)
					Mean_d[i],Std_d[i] = self.condmv_plotting_data(model,p_grid[i])
				## draw
				fig, axes = plt.subplots(ncols=4, figsize=(24, 5), constrained_layout=True)
				axes[0].plot(p_grid,Mean_t,linestyle='-',color='black')
				axes[0].plot(p_grid,Mean_d,linestyle='dashed',color='#6495ED')
				axes[0].set_title("Comparasion of Mean $E(\cdot|X_s)$, ite %d"%(epoch+1))
				axes[0].set_xlabel('$X_s$')
				axes[1].plot(p_grid,np.zeros(p_grid.shape),linestyle='-',color='black')
				axes[1].plot(p_grid,Mean_d-Mean_t,linestyle='dashed',color='#6495ED')
				axes[1].set_title("Error of Mean $E(\cdot|X_s)$, ite %d"%(epoch+1))
				axes[1].set_xlabel('$X_s$')
				axes[2].plot(p_grid,Std_t,linestyle='-',color='black')
				axes[2].plot(p_grid,Std_d,linestyle='dashed',color='red')
				axes[2].set_title("Comparasion of Std $Std(\cdot|X_s)$, ite %d"%(epoch+1))
				axes[2].set_xlabel('$X_s$')
				axes[3].plot(p_grid,np.zeros(p_grid.shape),linestyle='-',color='black')
				axes[3].plot(p_grid,Std_d-Std_t,linestyle='dashed',color='red')
				axes[3].set_title("Error of Std $Std(\cdot|X_s)$, ite %d"%(epoch+1))
				axes[3].set_xlabel('$X_s$')
				fig.savefig(path+'/cond_mvplot'+str(epoch+1)+'.png',dpi=150)
				plt.close()
			elif self.eqn_config.dim==2:
				# if self.eqn_config.eqn_name in ['MdOU']:
				# 	mzlim = [-3,3]
				# elif self.eqn_config.eqn_name in ['SO']:
				# 	mzlim = [-5.5,5.5]
				# else:
				# 	raise AttributeError('cond_meanvar: no this 2d distribution')
				## draw
				Npoint = self.monitor_config.cond_mv['Npoint']
				l1,l2 = self.monitor_config.cond_mv['range']
				p_gridx = np.linspace(l1[0],l1[1],Npoint+1)
				p_gridy = np.linspace(l2[0],l2[1],Npoint+1)
				p_gridx,p_gridy = np.meshgrid(p_gridx,p_gridy)
				p_grid = np.array((p_gridx.flatten(),p_gridy.flatten())).T
				PSh = p_grid.shape
				Mean_t, V_t, C_t = np.zeros(PSh),np.zeros(PSh),np.zeros(PSh[0])
				Mean_d, V_d, C_d = np.zeros(PSh),np.zeros(PSh),np.zeros(PSh[0])
				for i in range(p_grid.shape[0]):
					Mean_t[i],V_t[i],C_t[i] = self.condmv_plotting_std_cont2D(self.eqn_config.eqn_name,p_grid[i],self.Delta)
					Mean_d[i],V_d[i],C_d[i] = self.condmv_plotting_data2D(model,p_grid[i])
				## draw
				fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(24, 15), constrained_layout=True, subplot_kw={"projection": "3d"})
				# means
				axes[0,0].plot_surface(p_gridx, p_gridy, Mean_t[:,0].reshape([Npoint+1,Npoint+1]), cmap='Blues')
				axes[0,0].set_title("Truth Mean $E_1(\cdot|X_s)$, ite %d"%(epoch+1))
				axes[0,1].plot_surface(p_gridx, p_gridy, Mean_d[:,0].reshape([Npoint+1,Npoint+1]), cmap='Reds')
				axes[0,1].set_title("Estimated Mean $E_1(\cdot|X_s)$, ite %d"%(epoch+1))
				axes[0,2].plot_surface(p_gridx, p_gridy, Mean_t[:,1].reshape([Npoint+1,Npoint+1]), cmap='Blues')
				axes[0,2].set_title("Truth Mean $E_2(\cdot|X_s)$, ite %d"%(epoch+1))
				axes[0,3].plot_surface(p_gridx, p_gridy, Mean_d[:,1].reshape([Npoint+1,Npoint+1]), cmap='Reds')
				axes[0,3].set_title("Estimated Mean $E_2(\cdot|X_s)$, ite %d"%(epoch+1))
				axes[0,0].set_zlim([min(Mean_t[:,0]),max(Mean_t[:,0])])
				axes[0,1].set_zlim([min(Mean_t[:,0]),max(Mean_t[:,0])])
				axes[0,2].set_zlim([min(Mean_t[:,1]),max(Mean_t[:,1])])
				axes[0,3].set_zlim([min(Mean_t[:,1]),max(Mean_t[:,1])])
				# variances
				axes[1,0].plot_surface(p_gridx, p_gridy, V_t[:,0].reshape([Npoint+1,Npoint+1]), cmap='Blues')
				axes[1,0].set_title("Truth variance $Var_1(\cdot|X_s)$, ite %d"%(epoch+1))
				axes[1,1].plot_surface(p_gridx, p_gridy, V_d[:,0].reshape([Npoint+1,Npoint+1]), cmap='Reds')
				axes[1,1].set_title("Estimated variance $Var_1(\cdot|X_s)$, ite %d"%(epoch+1))
				axes[1,2].plot_surface(p_gridx, p_gridy, V_t[:,1].reshape([Npoint+1,Npoint+1]), cmap='Blues')
				axes[1,2].set_title("Truth variance $Var_2(\cdot|X_s)$, ite %d"%(epoch+1))
				axes[1,3].plot_surface(p_gridx, p_gridy, V_d[:,1].reshape([Npoint+1,Npoint+1]), cmap='Reds')
				axes[1,3].set_title("Estimated variance $Var_2(\cdot|X_s)$, ite %d"%(epoch+1))
				# covariance
				axes[2,0].plot_surface(p_gridx, p_gridy, C_t.reshape([Npoint+1,Npoint+1]), cmap='Blues')
				axes[2,0].set_title("Truth Covariance $Var_1(\cdot|X_s)$, ite %d"%(epoch+1))
				axes[2,1].plot_surface(p_gridx, p_gridy, C_d.reshape([Npoint+1,Npoint+1]), cmap='Reds')
				axes[2,1].set_title("Estimated Covariance $Var_1(\cdot|X_s)$, ite %d"%(epoch+1))
				fig.savefig(path+'/cond_mvplot'+str(epoch+1)+'.png',dpi=150)
				plt.close()
			else:
				pass
		else:
			pass

	def Ens_monitor(self,epoch,ckptmanager,ckp,model,DatVes):
		if not (set(['Testepoches_comp','Testepoches_endp']) <= set(dir(self))):
			self.Testepoches_comp,self.Testepoches_endp = self.Last_epoch_schedule(model.Testepoches)
		if (epoch+1) in self.Testepoches_comp:
			ckptmanager.save()
			# evaluate
			if ((epoch+1) in self.Testepoches_endp) or (epoch+1==self.N_epochs):
				modellist = self.readMultiplemodel(self.Ens_save_path,ckptmanager,model)
				if self.monitor_config.Ens_monitor['Ens_cond_mv']:
					self.cond_meanvar(modellist,epoch,enforce=True)
				if self.monitor_config.Ens_monitor['Ens_repdf']:
					self.complete_condpdf(modellist,epoch,enforce=True)
				if self.monitor_config.Ens_monitor['Ens_eva']:
					self.Eva_Ensemble(modellist,DatVes,epoch)
				if self.monitor_config.Ens_monitor['Ens_endpdf']:
					self.Endpdf_Ensemble(modellist,DatVes,epoch)

	def Eva_meanv_Multiple_last(self,model,epoch,DatVes,predt_path):
		if not (set(['Testepoches_comp','Testepoches_endp']) <= set(dir(self))):
			self.Testepoches_comp,self.Testepoches_endp = self.Last_epoch_schedule(model.Testepoches)
		if (epoch+1) in self.Testepoches_comp:
			if (epoch+1)==self.Testepoches_comp[0]:
				DatVes.test_mdat1model(model,predt_path,mode='w')
			else:
				DatVes.test_mdat1model(model,predt_path,mode='a')
			# evaluate
			if (epoch+1) in self.Testepoches_endp:
				self.Evaulation.plot_meancompare(save=True,epoch=('E'+str(epoch+1)))
				if ('Resdata' in self.eqn_config.keys()) and self.eqn_config.Resdata['if']:
					self.Evaulation.plot_meancompare_Resplus(save=True,epoch=('E'+str(epoch+1)))
				os.remove(predt_path)
		else:
			pass

	def fakesample_check(self,model,data,epoch):
		if (epoch+1==self.N_epochs) or ((epoch+1)%(int(self.N_epochs/self.monitor_config.fake_check['times']))==0) or (epoch==0):
			## data
			X_sample = self.fake_sample(model,data)
			## pca
			pca = sklearn.decomposition.PCA(n_components=2)
			X_new = pca.fit_transform(X_sample)
			d_new = pca.fit_transform(data)
			## draw
			fig, axes = plt.subplots(ncols=1, figsize=(8, 6), constrained_layout=True)
			axes.scatter(d_new[:,0],d_new[:,1], color='blue', alpha=0.2)
			axes.scatter(X_new[:,0],X_new[:,1], color='red', alpha=0.2)
			axes.set_title("PCA at ite %d"%(epoch+1))
			fig.savefig(self.pcapath+'/pca'+str(epoch+1)+'.png',dpi=200)
			plt.close()
			## fake sample test
			n_col = 8
			int_long = self.monitor_config.fake_check['int_long']
			n_row = model.dim*model.d_RNN//n_col+int(model.dim*model.d_RNN%n_col!=0)
			fig, axes = plt.subplots(nrows=n_row, ncols=n_col, figsize=(n_col*3, n_row*2), constrained_layout=True, squeeze=False)
			for i in range(n_row):
				for j in range(n_col):
					num = i*n_col+j
					if num<=(model.dim*model.d_RNN-1):
						axes[i,j].set_title("Place %d, ite %d"%(num+1,epoch+1))
						self.pdf_plotting_data(axes[i,j],data[:,num],int_long,'-','blue')
						self.pdf_plotting_data(axes[i,j],X_sample[:,num],int_long,'dashed','red')
					else:
						break
			fig.savefig(self.fakepdfpath+'/fakepdf'+str(epoch+1)+'.png',dpi=150)
			plt.close()
			## fake mean/variance test
			p_grid = np.arange(X_sample.shape[1])+1
			Mean_t, Mean_d = np.mean(data,axis=0),np.mean(X_sample,axis=0)
			Ecov_t, Ecov_d = -np.sort(-np.linalg.eig(np.cov((data.numpy()).T))[0]),-np.sort(-np.linalg.eig(np.cov(X_sample.T))[0])
			# draw
			fig, axes = plt.subplots(ncols=4, figsize=(24, 5), constrained_layout=True)
			axes[0].plot(p_grid,Mean_t,linestyle='-',color='black',marker='s')
			axes[0].plot(p_grid,Mean_d,linestyle='dashed',color='#6495ED',marker='s')
			axes[0].set_title("Comparasion of Mean of Generated data, ite %d"%(epoch+1))
			axes[0].set_xlabel('Component')
			axes[1].plot(p_grid,np.zeros(p_grid.shape),linestyle='-',color='black',marker='s')
			axes[1].plot(p_grid,Mean_d-Mean_t,linestyle='dashed',color='#6495ED',marker='s')
			axes[1].set_title("Error of Mean of Generated data, ite %d"%(epoch+1))
			axes[1].set_xlabel('Component')
			axes[2].plot(p_grid,Ecov_t,linestyle='-',color='black',marker='s')
			axes[2].plot(p_grid,Ecov_d,linestyle='dashed',color='red',marker='s')
			axes[2].set_yscale('log')
			axes[2].set_title("Comparasion of Spectrum of Correlation, ite %d"%(epoch+1))
			axes[2].set_xlabel('Component')
			axes[3].plot(p_grid,np.zeros(p_grid.shape),linestyle='-',color='black',marker='s')
			axes[3].plot(p_grid,Ecov_d-Ecov_t,linestyle='dashed',color='red',marker='s')
			axes[3].set_title("Error of Spectrum of Correlation, ite %d"%(epoch+1))
			axes[3].set_xlabel('Component')
			fig.savefig(self.fakemvpath+'/fake_mvplot'+str(epoch+1)+'.png',dpi=150)
			plt.close()
		else:
			pass

	def Eva_meanv(self,model,epoch,DatVes,predt_path):
		if (epoch+1==self.N_epochs) or ((epoch+1)%(int(self.N_epochs/self.monitor_config.Evameanv['times']))==0) or (epoch==0):
			DatVes.test_mdat1model(model,predt_path)
			self.Evaulation.plot_meancompare(save=True,epoch=('E'+str(epoch+1)))
			if ('Resdata' in self.eqn_config.keys()) and self.eqn_config.Resdata['if']:
				self.Evaulation.plot_meancompare_Resplus(save=True,epoch=('E'+str(epoch+1)))
		else:
			pass

	def Eva_loss(self,model,epoch,G_loss,D_loss):
		if ((epoch+1)%(int(self.N_epochs/self.monitor_config.loss['times']))==0):
			self.Evaulation.plot_train_hisGAN(self.N_epochs,G_loss,D_loss,savepath=(self.loss_path+'/loss.png'))
		else:
			pass

	def Eva_meanv_Multiple_last(self,model,epoch,DatVes,predt_path):
		if not (set(['Testepoches_comp','Testepoches_endp']) <= set(dir(self))):
			self.Testepoches_comp,self.Testepoches_endp = self.Last_epoch_schedule(model.Testepoches)
		if (epoch+1) in self.Testepoches_comp:
			if (epoch+1)==self.Testepoches_comp[0]:
				DatVes.test_mdat1model(model,predt_path,mode='w')
			else:
				DatVes.test_mdat1model(model,predt_path,mode='a')
			# evaluate
			if (epoch+1) in self.Testepoches_endp:
				self.Evaulation.plot_meancompare(save=True,epoch=('E'+str(epoch+1)))
				if ('Resdata' in self.eqn_config.keys()) and self.eqn_config.Resdata['if']:
					self.Evaulation.plot_meancompare_Resplus(save=True,epoch=('E'+str(epoch+1)))
				os.remove(predt_path)
		else:
			pass

	def Last_epoch_schedule(self,Testepoches):
		# schedule ensemble monitor test
		step = int(self.N_epochs/self.monitor_config.Evameanv['times'])
		end_p = step*(np.arange(int(self.N_epochs/step))+1)
		long = Testepoches[-1]-Testepoches[0]
		if step<long:
			raise AttributeError('Last_epoch_schedule: too many times for Eva_meanv_Multiple_last, please change to lower')
		# end_p = end_p[(end_p>long)*(end_p<Testepoches[0])]
		end_p = end_p[(end_p>long)]
		com_p = np.zeros(Testepoches.shape[0]*end_p.shape[0],dtype=int)
		for i in range(end_p.shape[0]):
			com_p[i*Testepoches.shape[0]:(i+1)*Testepoches.shape[0]] = Testepoches-(Testepoches[-1]-end_p[i])
		return com_p,end_p

	def fake_sample(self,model,data):
		Xs = data[:,:model.dim]
		X_sample = [Xs] 
		for i in range(model.d_RNN-1):
			Z_samp = tf.random.normal([data.shape[0], model.n_Z])
			Xs = model.Generate(Xs,Z_samp)
			X_sample += [Xs]
		X_sample = np.concatenate(X_sample, -1)
		return X_sample

	def condmv_plotting_data(self,model,x,N=5000):
		try:
			data = (model.predict(np.repeat(x,N)[:,None])).numpy().flatten()
		except:
			Nmodel = len(model)
			modelsep = np.linspace(0,N,Nmodel+1,dtype=int)
			data = np.zeros(N)
			for i in range(Nmodel):
				data[modelsep[i]:modelsep[i+1]] = (model[i].predict(np.repeat(x,modelsep[i+1]-modelsep[i])[:,None])).numpy().flatten()
		m,s = np.mean(data),np.std(data)
		return m,s

	def condpdf_plotting_std(self,name,ax,intlong,x,Delta):
		if name=='Brownian Motion':
			x_axis = np.linspace(x-intlong/2,x+intlong/2,200)
			ax.plot(x_axis, scipy.stats.norm.pdf(x_axis, x, np.sqrt(Delta)),color='#000080',label='Reference')
		elif name=='Geometric Brownian Motion':
			x_axis = np.linspace(x*np.exp(self.eqn_config.mu*Delta)-intlong/2*x/3,x*np.exp(self.eqn_config.mu*Delta)+intlong/2*x/3,200)
			cgeobw = (self.eqn_config.mu-(self.eqn_config.sigma**2)/2)
			GeoBpdf = np.zeros(x_axis.shape)
			_id = (x_axis>0)*(np.abs(x_axis)>1.0e-9)
			GeoBpdf[_id] = np.exp(-(np.log(x_axis[_id]/x)-cgeobw*Delta)**2/(2*self.eqn_config.sigma**2*Delta))/(np.sqrt(2*np.pi*Delta)*self.eqn_config.sigma*x_axis[_id])
			ax.plot(x_axis, GeoBpdf,color='#000080',label='Reference')
		elif name=='OU Process':
			x_axis = np.linspace(x-intlong/2,x+intlong/2,200)
			mea = self.eqn_config.mu+(x-self.eqn_config.mu)*np.exp(-self.eqn_config.theta*Delta)
			var = self.eqn_config.sigma**2/(2*self.eqn_config.theta)*(1-np.exp(-2*self.eqn_config.theta*Delta))
			ax.plot(x_axis, scipy.stats.norm.pdf(x_axis, mea, np.sqrt(var)),color='#000080',label='Reference')
		# elif name=='Exp_diffusion':
		# 	x_axis = np.linspace(x-intlong/2,x+intlong/2,200)
		# 	mapsample = self.cond_sample_EM(name,x,Delta)
		# 	kde = scipy.stats.kde.gaussian_kde(mapsample)
		# 	ax.plot(x_axis, kde(x_axis), color='#000080',label='Reference')
		elif name=='Exp_diffusion':
			x_axis = np.linspace(x-intlong/2,x+intlong/2,200)
			mea = x-self.eqn_config.mu*x*Delta
			std = self.eqn_config.sigma*np.exp(-x**2)*np.sqrt(Delta)
			ax.plot(x_axis, scipy.stats.norm.pdf(x_axis, mea, std),color='#000080',label='Reference')
		# elif name=='Trig_drift':
		# 	x_axis = np.linspace(x-intlong/2,x+intlong/2,200)
		# 	mapsample = self.cond_sample_EM(name,x,Delta)
		# 	kde = scipy.stats.kde.gaussian_kde(mapsample)
		# 	ax.plot(x_axis, kde(x_axis), color='#000080',label='Reference')
		elif name=='Trig_drift':
			x_axis = np.linspace(x-intlong/2,x+intlong/2,200)
			mea = x+np.sin(2*self.eqn_config.k*np.pi*x)*Delta
			std = abs(self.eqn_config.sigma*np.cos(2*self.eqn_config.k*np.pi*x))*np.sqrt(Delta)
			ax.plot(x_axis, scipy.stats.norm.pdf(x_axis, mea, std),color='#000080',label='Reference')
		elif name=='Exp_OU':
			x_axis = np.linspace(x-intlong/2,x+intlong/2,200)
			x_axis = x_axis[x_axis>0]
			th,dt  = self.eqn_config.theta,self.eqn_config.Delta
			mu,sig = self.eqn_config.mu,self.eqn_config.sigma
			MU,SIG = (1-th*dt)*np.log(x)+th*mu*dt,sig*np.sqrt(dt)
			pdf = 1/(x_axis*SIG*np.sqrt(2*np.pi))*np.exp(-(np.log(x_axis)-MU)**2/(2*SIG**2))
			ax.plot(x_axis, pdf, color='#000080',label='Reference')
		# elif name=='Double_well':
		# 	x_axis = np.linspace(x-intlong/2,x+intlong/2,200)
		# 	mapsample = self.cond_sample_EM(name,x,Delta)
		# 	kde = scipy.stats.kde.gaussian_kde(mapsample)
		# 	ax.plot(x_axis, kde(x_axis), color='#000080',label='Reference')
		elif name=='Double_well':
			x_axis = np.linspace(x-intlong/2,x+intlong/2,200)
			mea = x+(x-x**3)*Delta
			std = self.eqn_config.sigma*np.sqrt(Delta)
			ax.plot(x_axis, scipy.stats.norm.pdf(x_axis, mea, std),color='#000080',label='Reference')
		elif name=='Exp_dis':
			x_axis = np.linspace(x-intlong/2,x+intlong/2,200)
			a = x+self.eqn_config.theta*x*self.eqn_config.Delta
			b = self.eqn_config.sigma*np.sqrt(self.eqn_config.Delta)
			_id = x_axis>=a
			x_plot = x_axis[_id]
			pdf_plot = np.exp(-(x_axis[_id]-a)/b)/b
			ax.plot(x_plot, pdf_plot,color='#000080',label='Reference')
			ax.plot([a,a], [0,1/b],color='#000080')
		else:
			print('The distribution %s is not supported'%(name))

	def condpdf_plotting_std2D(self,name,ax,intlong,x,Delta,level):
		if name=='MdOU':
			Mean = np.array(x)+np.dot(np.array(self.eqn_config.mu),np.array(x))*Delta
			Cov = (np.array(self.eqn_config.sigma).T).dot(np.array(self.eqn_config.sigma))*Delta
			distx = np.linspace(x[0]-intlong[0]/2,x[0]+intlong[0]/2,200)
			disty = np.linspace(x[1]-intlong[1]/2,x[1]+intlong[1]/2,200)
			distx,disty = np.meshgrid(distx,disty)
			rv = scipy.stats.multivariate_normal(Mean, Cov)
			f = rv.pdf(np.dstack((distx,disty)))
			cfset = ax.contourf(distx, disty, f, cmap='Blues')
			cset = ax.contour(distx, disty, f, colors='k')
			ax.clabel(cset, inline=1, fontsize=7)
		elif name=='SO':
			pass
		else:
			print('The distribution %s is not supported'%(name))

	def condpdf_combine_plotting_std2D(self,name,intlong,x,Delta,level):
		if name=='MdOU':
			Mean = np.array(x)+np.dot(np.array(self.eqn_config.mu),np.array(x))*Delta
			Cov = (np.array(self.eqn_config.sigma).T).dot(np.array(self.eqn_config.sigma))*Delta
			data = np.random.multivariate_normal(Mean, Cov, size=500000)
			xlimit = [x[0]-intlong[0]/1.4,x[0]+intlong[0]/1.4]
			ylimit = [x[1]-intlong[1]/1.4,x[1]+intlong[1]/1.4]
			a = sns.jointplot(x=data[:,0], y=data[:,1], fill=True, kind="kde", color="#004C99", levels=level, xlim=xlimit,ylim=ylimit, height=6)
			font2 = {'size'   : 14,}
			patch = matplotlib.patches.Patch(color='#004C99', alpha=0.3, label='Reference')
			plt.legend(handles=[patch],prop=font2)
			return a
		elif name=='SO':
			pass
		else:
			print('The distribution %s is not supported'%(name))

	def condmargpdf_plotting_std2D(self,name,ax1,ax2,intlong,x,Delta):
		if name=='MdOU':
			Mean = np.array(x)+np.dot(np.array(self.eqn_config.mu),np.array(x))*Delta
			Cov = (np.array(self.eqn_config.sigma).T).dot(np.array(self.eqn_config.sigma))*Delta
			distx = np.linspace(x[0]-intlong[0]/2,x[0]+intlong[0]/2,200)
			disty = np.linspace(x[1]-intlong[1]/2,x[1]+intlong[1]/2,200)
			ax1.plot(distx, scipy.stats.norm.pdf(distx, Mean[0], np.sqrt(Cov[0,0])),color='#000080',label='Reference')
			ax2.plot(disty, scipy.stats.norm.pdf(disty, Mean[1], np.sqrt(Cov[1,1])),color='#000080',label='Reference')
			# rv = scipy.stats.multivariate_normal(Mean, Cov)
			# f = rv.pdf(np.dstack((distx,disty)))
			# cfset = ax.contourf(distx, disty, f, cmap='Blues')
			# cset = ax.contour(distx, disty, f, colors='k')
			# ax.clabel(cset, inline=1, fontsize=7)
		elif name=='SO':
			Mean = np.array(x)+np.dot(np.array(self.eqn_config.mu),np.array(x))*Delta
			Cov = (np.array(self.eqn_config.sigma).T).dot(np.array(self.eqn_config.sigma))*Delta
			distx = np.linspace(x[0]-intlong[0]/5,x[0]+intlong[0]/5,200)
			disty = np.linspace(x[1]-intlong[1]/2,x[1]+intlong[1]/2,200)
			## ax1
			# ax1.plot([Mean[0]],[0],color='#000080',label='Reference', marker=".", markersize=25)
			## alternate for ax1
			ax1.set_ylim([-10,540])
			ax1.annotate("",xy=(Mean[0], 520), xycoords='data',xytext=(Mean[0], 0), textcoords='data',arrowprops=dict(arrowstyle="-|>, head_width=0.1",mutation_scale=30,connectionstyle="arc3",color='#000080'),)
			ax1.plot(distx,np.zeros(distx.shape),color='#000080')
			ax1.plot([Mean[0]],[0],color='#000080',label='Reference', marker='o', markerfacecolor='white', markersize=8)
			ax1.set_xlim([x[0]-intlong[0]/10,x[0]+intlong[0]/10])
			## ax2
			ax2.plot(disty, scipy.stats.norm.pdf(disty, Mean[1], np.sqrt(Cov[1,1])),color='#000080',label='Reference')
			ax2.set_xlim([x[1]-intlong[0]/8,x[1]+intlong[0]/8])
		else:
			print('The distribution %s is not supported'%(name))

	def condpdf_plotting_data(self,name,ax,model,intlong,x,Delta,N=10000):
		try:
			data = (model.predict(np.repeat(x,N)[:,None])).numpy().flatten()
		except:
			Nmodel = len(model)
			modelsep = np.linspace(0,N,Nmodel+1,dtype=int)
			data = np.zeros(N)
			for i in range(Nmodel):
				data[modelsep[i]:modelsep[i+1]] = (model[i].predict(np.repeat(x,modelsep[i+1]-modelsep[i])[:,None])).numpy().flatten()
		kde = scipy.stats.kde.gaussian_kde(data)
		if name=='Brownian Motion':
			dist_space = np.linspace(x-intlong/2,x+intlong/2,200)
		elif name=='Geometric Brownian Motion':
			dist_space = np.linspace(x-intlong/2*x/3,x+intlong/2*x/3,200)
		elif name=='OU Process':
			m = self.eqn_config.mu+(x-self.eqn_config.mu)*np.exp(-self.eqn_config.theta*Delta)
			dist_space = np.linspace(m-intlong/2,m+intlong/2,200)
		elif name=='Exp_diffusion':
			dist_space = np.linspace(x-intlong/2,x+intlong/2,200)
		elif name=='Trig_drift':
			dist_space = np.linspace(x-intlong/2,x+intlong/2,200)
		elif name=='Exp_OU':
			dist_space = np.linspace(x-intlong/2,x+intlong/2,200)
			dist_space = dist_space[dist_space>0]
		elif name=='Double_well':
			dist_space = np.linspace(x-intlong/2,x+intlong/2,200)
		elif name=='Exp_dis':
			dist_space = np.linspace(x-intlong/2,x+intlong/2,200)
		else:
			print('The distribution %s is not supported'%(name))
		# pdb.set_trace()
		# ax.plot(dist_space,kde(dist_space),linestyle='dashed',color='red')
		ax.hist(data, bins=50, alpha=0.6, ec="k", color='#A0A0A0', density=True, histtype='stepfilled',label='Learned')

	def condpdf_plotting_data2D(self,name,ax,model,intlong,x,Delta,level,N=10000):
		try:
			data = (model.predict(np.tile(x,[N,1]))).numpy()
		except:
			Nmodel = len(model)
			modelsep = np.linspace(0,N,Nmodel+1,dtype=int)
			data = np.zeros([N,2])
			for i in range(Nmodel):
				data[modelsep[i]:modelsep[i+1]] = (model[i].predict(np.tile(x,[modelsep[i+1]-modelsep[i],1]))).numpy()
		kde = scipy.stats.kde.gaussian_kde(data.T)
		if name in ['MdOU','SO']:
			distx = np.linspace(x[0]-intlong[0]/2,x[0]+intlong[0]/2,200)
			disty = np.linspace(x[1]-intlong[1]/2,x[1]+intlong[1]/2,200)
			distx,disty = np.meshgrid(distx,disty)
		else:
			print('The distribution %s is not supported'%(name))
		# pdb.set_trace()
		f = np.reshape(kde(np.vstack([distx.ravel(), disty.ravel()])), distx.shape)
		cfset = ax.contourf(distx, disty, f, cmap='Reds')
		cset = ax.contour(distx, disty, f, colors='k')
		ax.clabel(cset, inline=1, fontsize=7)

	def condpdf_combine_plotting_data2D(self,name,model,intlong,x,Delta,level,N=10000):
		try:
			data = (model.predict(np.tile(x,[N,1]))).numpy()
		except:
			Nmodel = len(model)
			modelsep = np.linspace(0,N,Nmodel+1,dtype=int)
			data = np.zeros([N,2])
			for i in range(Nmodel):
				data[modelsep[i]:modelsep[i+1]] = (model[i].predict(np.tile(x,[modelsep[i+1]-modelsep[i],1]))).numpy()
		if name in ['MdOU','SO']:
			xlimit = [x[0]-intlong[0]/1.4,x[0]+intlong[0]/1.4]
			ylimit = [x[1]-intlong[1]/1.4,x[1]+intlong[1]/1.4]
		else:
			print('The distribution %s is not supported'%(name))
		# pdb.set_trace()
		a = sns.jointplot(x=data[:,0], y=data[:,1], fill=True, kind="kde", color="#990000", levels=level, xlim=xlimit,ylim=ylimit, height=6)
		font2 = {'size'   : 14,}
		patch = matplotlib.patches.Patch(color='#990000', alpha=0.3, label='Learned')
		plt.legend(handles=[patch],prop=font2)
		return a

	def condmargpdf_plotting_data2D(self,name,ax1,ax2,model,intlong,x,Delta,N=10000):
		try:
			data = (model.predict(np.tile(x,[N,1]))).numpy()
		except:
			Nmodel = len(model)
			modelsep = np.linspace(0,N,Nmodel+1,dtype=int)
			data = np.zeros([N,2])
			for i in range(Nmodel):
				data[modelsep[i]:modelsep[i+1]] = (model[i].predict(np.tile(x,[modelsep[i+1]-modelsep[i],1]))).numpy()
		if name=='MdOU':
			distx = np.linspace(x[0]-intlong[0]/2,x[0]+intlong[0]/2,200)
			disty = np.linspace(x[1]-intlong[1]/2,x[1]+intlong[1]/2,200)
		elif name=='SO':
			distx = np.linspace(x[0]-intlong[0]/5,x[0]+intlong[0]/5,200)
			disty = np.linspace(x[1]-intlong[1]/2,x[1]+intlong[1]/2,200)
		else:
			print('The distribution %s is not supported'%(name))
		kde = scipy.stats.kde.gaussian_kde(data[:,0])
		## ax1
		ax1.plot(distx,kde(distx),color='#DC143C',linestyle='dashed',label='Learned')
		# ax1.hist(data[:,0], bins=50, alpha=0.6, ec="k", density=True, histtype='stepfilled',label='Learned')
		## ax2
		ax2.hist(data[:,1], bins=50, alpha=0.6, ec="k", color='#A0A0A0', density=True, histtype='stepfilled',label='Learned')

	def condmv_plotting_std(self,name,x,Delta):
		if name=='Brownian Motion':
			m,s = x,np.sqrt(Delta)
			return m,s
		elif name=='Geometric Brownian Motion':
			m,s = x*np.exp(self.eqn_config.mu*Delta),np.abs(x*np.exp(self.eqn_config.mu*Delta)*np.sqrt(np.exp(self.eqn_config.sigma**2*Delta)-1))
			return m,s
		elif name=='OU Process':
			# m   = self.eqn_config.mu+(x-self.eqn_config.mu)*np.exp(-self.eqn_config.theta*Delta)
			m   = self.eqn_config.theta*(self.eqn_config.mu-x)
			var = self.eqn_config.sigma**2/(2*self.eqn_config.theta)*(1-np.exp(-2*self.eqn_config.theta*Delta))
			return m,np.sqrt(var)
		elif name=='Exp_diffusion':
			mapsample = self.cond_sample_EM(name,x,Delta)
			m   = np.mean(mapsample)
			var = np.var(mapsample)
			return m,np.sqrt(var)
		elif name=='Trig_drift':
			mapsample = self.cond_sample_EM(name,x,Delta)
			m   = np.mean(mapsample)
			var = np.var(mapsample)
			return m,np.sqrt(var)
		elif name=='Exp_OU':
			th,dt  = self.eqn_config.theta,Delta
			mu,sig = self.eqn_config.mu,self.eqn_config.sigma
			MU,SIG = (1-th*dt)*np.log(x)+th*mu*dt,sig*np.sqrt(dt)
			m = np.exp(MU+SIG**2/2)
			var = (np.exp(SIG**2)-1)*np.exp(2*MU+SIG**2)
			return m,np.sqrt(var)
		elif name=='Double_well':
			mapsample = self.cond_sample_EM(name,x,Delta)
			m   = np.mean(mapsample)
			var = np.var(mapsample)
			return m,np.sqrt(var)
		elif name=='Exp_dis':
			m = x+self.eqn_config.theta*x*self.eqn_config.Delta+self.eqn_config.sigma*np.sqrt(self.eqn_config.Delta)
			std = self.eqn_config.sigma*np.sqrt(self.eqn_config.Delta)
			return m,std
		else:
			print('The distribution %s is not supported'%(name))

	def condmv_plotting_std_cont(self,name,x,Delta):
		if name=='Brownian Motion':
			m,s = x,np.sqrt(Delta)
			return m,s
		elif name=='Geometric Brownian Motion':
			m,s = self.eqn_config.mu*x,np.abs(self.eqn_config.sigma*x)
			return m,s
		elif name=='OU Process':
			m   = self.eqn_config.theta*(self.eqn_config.mu-x)
			var = self.eqn_config.sigma**2
			return m,np.sqrt(var)
		elif name=='Exp_diffusion':
			m   = -self.eqn_config.mu*x
			std = self.eqn_config.sigma*np.exp(-x**2)
			return m,std
		elif name=='Trig_drift':
			m   = np.sin(2*self.eqn_config.k*np.pi*x)
			std = abs(self.eqn_config.sigma*np.cos(2*self.eqn_config.k*np.pi*x))
			return m,std
		elif name=='Exp_OU':
			th,dt  = self.eqn_config.theta,Delta
			mu,sig = self.eqn_config.mu,self.eqn_config.sigma
			MU,SIG = (1-th*dt)*np.log(x)+th*mu*dt,sig*np.sqrt(dt)
			m = -th*np.log(x)+th*mu+sig**2/2
			var = (np.exp(SIG**2)-1)*np.exp(2*MU+SIG**2)
			return m,np.sqrt(var)
		elif name=='Double_well':
			m   = x-x**3
			std = self.eqn_config.sigma
			return m,std
		elif name=='Exp_dis':
			m = self.eqn_config.theta*x+self.eqn_config.sigma/np.sqrt(self.eqn_config.Delta)
			std = self.eqn_config.sigma
			return m,std
		else:
			print('The distribution %s is not supported'%(name))

	def condmv_plotting_std_cont2D(self,name,x,Delta):
		if name in ['MdOU','SO']:
			m = np.dot(np.array(self.eqn_config.mu),np.array((x)))
			s = np.array(self.eqn_config.sigma)
			cov = (s.T).dot(s)*Delta
			return m,np.diagonal(cov),cov[0,1]
		else:
			print('The distribution %s is not supported'%(name))

	def cond_sample_EM(self,equname,x,Delta):
		N = 10000
		if equname=='Exp_diffusion':
			xt = x-self.eqn_config.mu*x*Delta+self.eqn_config.sigma*np.exp(-x**2)*np.random.normal(0.0, np.sqrt(Delta), N)
		elif equname=='Trig_drift':
			xt = x+np.sin(2*self.eqn_config.k*np.pi*x)*Delta+self.eqn_config.sigma*np.cos(2*self.eqn_config.k*np.pi*x)*np.random.normal(0.0, np.sqrt(Delta), N)
		elif equname=='Double_well':
			xt = x+(x-x**3)*Delta+self.eqn_config.sigma*np.random.normal(0.0, np.sqrt(Delta), N)
		else:
			print('The distribution %s is not supported'%(name))
		return xt

	def condmv_plotting_data(self,model,x,N=5000):
		try:
			data = (model.predict(np.repeat(x,N)[:,None])).numpy().flatten()
		except:
			Nmodel = len(model)
			modelsep = np.linspace(0,N,Nmodel+1,dtype=int)
			data = np.zeros(N)
			for i in range(Nmodel):
				data[modelsep[i]:modelsep[i+1]] = (model[i].predict(np.repeat(x,modelsep[i+1]-modelsep[i])[:,None])).numpy().flatten()
		m,s = np.mean(data),np.std(data)
		return m,s

	def condmv_plotting_data2D(self,model,x,N=5000):
		try:
			data = (model.predict(np.tile(x,[N,1]))).numpy()
		except:
			Nmodel = len(model)
			modelsep = np.linspace(0,N,Nmodel+1,dtype=int)
			data = np.zeros([N,2])
			for i in range(Nmodel):
				data[modelsep[i]:modelsep[i+1]] = (model[i].predict(np.tile(x,[modelsep[i+1]-modelsep[i],1]))).numpy()
		m,s = (np.mean(data,axis=0)-x)/self.eqn_config.Delta,np.cov(data.T)
		return m,np.diagonal(s),s[0,1]

	def pdf_plotting_data(self,ax,data,intlong,line_,color_,label_='Ground Truth'):
		m = np.mean(data)
		kde = scipy.stats.kde.gaussian_kde(data)
		dist_space = np.linspace(m-intlong/2,m+intlong/2,200)
		ax.plot(dist_space,kde(dist_space),linestyle=line_,color=color_,label=label_)

	def data2dhistogram(self,data,Delta,name):
		# data should in the form of [num of trajectory, trajectory]
		N_data, Trac_long = data.shape
		x = np.arange(Trac_long)*Delta
		# fine the data
		num_fine = 800
		x_fine = np.linspace(x.min(), x.max(), num_fine)
		y_fine = np.empty((N_data, num_fine), dtype=float)
		for i in range(N_data):
			y_fine[i, :] = np.interp(x_fine, x, data[i, :])
		data_ = y_fine.flatten()
		x_ = np.tile(x_fine, N_data)
		# draw
		fig, ax = plt.subplots(ncols=1, figsize=(12, 4), constrained_layout=True)
		h, xedges, yedges = np.histogram2d(x_, data_, bins=[400, 100])
		pcm = ax.pcolormesh(xedges, yedges, h.T, cmap=plt.cm.plasma, vmax=np.max(h), rasterized=True)
		# pcm = ax.pcolormesh(xedges, yedges, h.T, cmap=plt.cm.plasma, norm=matplotlib.colors.LogNorm(vmax=5.5e2), rasterized=True)
		fig.colorbar(pcm, ax=ax, label="# points", pad=0)
		ax.set_title("Hit Diagram of %s"%(name))
		ax.set_xlabel("Time")
		ax.set_ylabel("Value of Data")
		fig.savefig(self.dataplotpath+'/hist_'+name+'.png',dpi=250)
		plt.close()

	def transprobinfo(self,data,name):
		# data should in the form of [num of trajectory, trajectory]
		data_in = (data[:,:-1]).flatten()
		fig, ax = plt.subplots(ncols=1, figsize=(12, 4), constrained_layout=True)
		ax.hist(data_in, bins=50, alpha=0.6, ec="k", histtype='stepfilled')
		ax.set_title("Number of trasition input data $X_s$ from %s"%(name))
		ax.set_xlabel("Value of $X_s$")
		ax.set_ylabel("# of data")
		ax.grid(alpha=0.7)
		fig.savefig(self.dataplotpath+'/hist_input_'+name+'.png',dpi=250)
		plt.close()

	def readMultiplemodel(self,path,ckptmanager,model):
		# This function is designed for test for multiple models
		modellist = []
		modeldict = ckptmanager.checkpoints
		for i in range(len(modeldict)):
			print(i)
			ModelX = self.GANsModel(self.config)
			# see https://www.tensorflow.org/api_docs/python/tf/train/Checkpoint
			ModelXCheckp = tf.train.Checkpoint(G_optimizer=ModelX.G_optimizer,D_optimizer=ModelX.D_optimizer,G=ModelX.G,D=ModelX.D)
			# manager = tf.train.CheckpointManager(ModelXCheckp, path, max_to_keep=1)
			ModelXCheckp.restore(modeldict[i]).expect_partial()
			modellist.append(ModelX)
		return modellist

	#---------------------------------- below is predictor --------------------------------
	def Ens_predictor(self,epoch,ckptmanager,ckp,model,DatVes,test_data,savepath):
		# path
		if not os.path.exists(savepath):
			os.makedirs(savepath)
		# evaluate
		modellist = self.readMultiplemodel(self.Ens_save_path,ckptmanager,model)
		# self.plot_sampletraindata(DatVes,savepath)
		self.cond_meanvar_Enspre(modellist,epoch,savepath,cont=True)
		# self.cond_dis_Enspre(modellist,epoch,savepath)
		self.Eva_Enspre(modellist,epoch,savepath,DatVes,test_data=test_data)
		# self.fakemeanvar_Enspre(modellist,DatVes.train_mat,epoch,savepath)

	def cond_meanvar_Enspre(self,model,epoch,savepath,cont=False):
		## compute
		# Npoint = self.monitor_config.cond_mv['Npoint']
		# l1,l2 = self.monitor_config.cond_mv['range']
		if self.eqn_config.dim==1:
			Npoint = 80
			# condition mv functions
			if cont:
				condmvfunc = self.condmv_plotting_std_cont
			else:
				condmvfunc = self.condmv_plotting_std
			# upper and lower limits
			if self.eqn_config.eqn_name=='Geometric Brownian Motion':
				l1,l2 = 0.5,15
			elif self.eqn_config.eqn_name=='OU Process':
				l1,l2 = 0.7,2
			elif self.eqn_config.eqn_name=='Exp_diffusion':
				l1,l2 = -0.6,0.6
			elif self.eqn_config.eqn_name=='Trig_drift':
				l1,l2 = 0.3,0.7
			elif self.eqn_config.eqn_name=='Exp_OU':
				l1,l2 = 0.25,1.7
			elif self.eqn_config.eqn_name=='Double_well':
				l1,l2 = -2,2
				# l1,l2 = -1.8,1.8
			elif self.eqn_config.eqn_name=='Exp_dis':
				l1,l2 = 0.3,0.9
			p_grid = np.linspace(l1,l2,Npoint+1)
			Mean_t, Std_t = np.zeros(p_grid.shape),np.zeros(p_grid.shape)
			Mean_d, Std_d = np.zeros(p_grid.shape),np.zeros(p_grid.shape)
			for i in range(p_grid.shape[0]):
				Mean_t[i],Std_t[i] = condmvfunc(self.eqn_config.eqn_name,p_grid[i],self.Delta)
				Mean_d[i],Std_d[i] = self.condmv_plotting_data(model,p_grid[i],N=500000)
			## change scale
			if self.eqn_config.eqn_name=='Exp_OU':
				Mean_d = (np.log(Mean_d)-np.log(p_grid))/self.eqn_config.Delta
			else:
				Mean_d = (Mean_d-p_grid)/self.eqn_config.Delta
				Std_d = Std_d/np.sqrt(self.eqn_config.Delta)
			
			if not cont:
				if self.eqn_config.eqn_name=='Exp_OU':
					pass
				else:
					Mean_t = (Mean_t-p_grid)/self.eqn_config.Delta
					Mean_d = (Mean_d-p_grid)/self.eqn_config.Delta
					Std_t = Std_t/np.sqrt(self.eqn_config.Delta)
					Std_d = Std_d/np.sqrt(self.eqn_config.Delta)
			else:
				pass
			## draw
			font2 = {'size'   : 14,}
			# drift
			fig1,ax1 = plt.subplots(figsize=(6,4))
			ax1.plot(p_grid,Mean_t,linestyle='-', linewidth=2.0,color='#000080',label='Reference')
			ax1.plot(p_grid,Mean_d,linestyle='dashed', linewidth=2.0,color='#DC143C',label='Learned')
			ax1.set_xlabel('x', font2)
			ax1.set_ylabel('a(x)', font2)
			ax1.legend(prop=font2)
			# print('Drift Error: '+"{:.4E}".format(np.sqrt((p_grid[2]-p_grid[1]))*np.linalg.norm(Mean_t-Mean_d)))
			print('Drift Error: '+"{:.4E}".format(np.linalg.norm(Mean_t-Mean_d)/np.linalg.norm(Mean_t)))
			# diffusion
			fig2,ax2 = plt.subplots(figsize=(6,4))
			ax2.plot(p_grid,Std_t,linestyle='-', linewidth=2.0,color='#000080',label='Reference')
			ax2.plot(p_grid,Std_d,linestyle='dashed', linewidth=2.0,color='#DC143C',label='Learned')
			ax2.set_xlabel('x', font2)
			ax2.set_ylabel('b(x)', font2)
			# ax2.set_ylim([0.09,0.11])
			std_low = min(np.min(Std_t),np.min(Std_d))
			std_high = max(np.max(Std_t),np.max(Std_d))
			std_pad = 0.1*(std_high-std_low) if std_high>std_low else 0.1*abs(std_high)
			ax2.set_ylim([std_low-std_pad,std_high+std_pad])
			ax2.legend(prop=font2)
			# print('Diffusion Error: '+"{:.4E}".format(np.sqrt((p_grid[2]-p_grid[1]))*np.linalg.norm(Std_t-Std_d)))
			print('Diffusion Error: '+"{:.4E}".format(np.linalg.norm(Std_t-Std_d)/np.linalg.norm(Std_t)))
			## save
			fig1.savefig(savepath+'/condmean.pdf', bbox_inches='tight')
			fig2.savefig(savepath+'/condstd.pdf', bbox_inches='tight')
		elif self.eqn_config.dim==2:
			Npoint = 20
			# upper and lower limits
			if self.eqn_config.eqn_name=='MdOU':
				l1,l2 = [-2.0,2.0],[-1.0,1.0]
			elif self.eqn_config.eqn_name=='SO':
				l1,l2 = [-2.0,2.0],[-1.0,1.0]
			p_gridx = np.linspace(l1[0],l1[1],Npoint+1)
			p_gridy = np.linspace(l2[0],l2[1],Npoint+1)
			p_gridx,p_gridy = np.meshgrid(p_gridx,p_gridy)
			p_grid = np.array((p_gridx.flatten(),p_gridy.flatten())).T
			PSh = p_grid.shape
			Mean_t, V_t, C_t = np.zeros(PSh),np.zeros(PSh),np.zeros(PSh[0])
			Mean_d, V_d, C_d = np.zeros(PSh),np.zeros(PSh),np.zeros(PSh[0])
			for i in range(p_grid.shape[0]):
				Mean_t[i],V_t[i],C_t[i] = self.condmv_plotting_std_cont2D(self.eqn_config.eqn_name,p_grid[i],self.Delta)
				Mean_d[i],V_d[i],C_d[i] = self.condmv_plotting_data2D(model,p_grid[i],N=500000)

			V_t,V_d = np.sqrt(V_t/self.eqn_config.Delta),np.sqrt(V_d/self.eqn_config.Delta)
			C_t,C_d = C_t/self.eqn_config.Delta,C_d/self.eqn_config.Delta
			
			font2 = {'size'   : 14,}
			# drift
			fig1, ax1 = plt.subplots(ncols=4, figsize=[24,4])
			min1 = min(Mean_t[:,0].min(),Mean_d[:,0].min())
			max1 = max(Mean_t[:,0].max(),Mean_d[:,0].max())
			min2 = min(Mean_t[:,1].min(),Mean_d[:,1].min())
			max2 = max(Mean_t[:,1].max(),Mean_d[:,1].max())
			r1,r2 = np.linspace(min1,max1,30),np.linspace(min2,max2,30)
			pgx,pgy = p_grid[:,0].reshape([Npoint+1,Npoint+1]),p_grid[:,1].reshape([Npoint+1,Npoint+1])
			ax1[0].contour(pgx,pgy,Mean_t[:,0].reshape([Npoint+1,Npoint+1]), colors='k',       vmin=min1, vmax=max1, levels=r1)
			ax1[1].contour(pgx,pgy,Mean_d[:,0].reshape([Npoint+1,Npoint+1]), colors='#DC143C', vmin=min1, vmax=max1, levels=r1)
			ax1[2].contour(pgx,pgy,Mean_t[:,1].reshape([Npoint+1,Npoint+1]), colors='k',       vmin=min2, vmax=max2, levels=r2)
			ax1[3].contour(pgx,pgy,Mean_d[:,1].reshape([Npoint+1,Npoint+1]), colors='#DC143C', vmin=min2, vmax=max2, levels=r2)
			fig1.savefig(savepath+'/condmeancontour.pdf')

			# means
			fig, axes = plt.subplots(ncols=4, figsize=(24, 4), constrained_layout=True, subplot_kw={"projection": "3d"})
			self.whitebg(axes)
			axes[0].plot_surface(p_gridx, p_gridy, Mean_t[:,0].reshape([Npoint+1,Npoint+1]), cmap='Blues')
			axes[1].plot_surface(p_gridx, p_gridy, Mean_d[:,0].reshape([Npoint+1,Npoint+1]), cmap='Reds')
			axes[2].plot_surface(p_gridx, p_gridy, Mean_t[:,1].reshape([Npoint+1,Npoint+1]), cmap='Blues')
			axes[3].plot_surface(p_gridx, p_gridy, Mean_d[:,1].reshape([Npoint+1,Npoint+1]), cmap='Reds')
			axes[0].set_zlim([min(Mean_t[:,0]),max(Mean_t[:,0])])
			axes[1].set_zlim([min(Mean_t[:,0]),max(Mean_t[:,0])])
			axes[2].set_zlim([min(Mean_t[:,1]),max(Mean_t[:,1])])
			axes[3].set_zlim([min(Mean_t[:,1]),max(Mean_t[:,1])])
			fig.savefig(savepath+'/condmean.pdf')
			# variances
			fig, axes = plt.subplots(ncols=4, figsize=(24, 4), constrained_layout=True, subplot_kw={"projection": "3d"})
			self.whitebg(axes)
			axes[0].plot_surface(p_gridx, p_gridy, V_t[:,0].reshape([Npoint+1,Npoint+1]), cmap='Blues')
			axes[1].plot_surface(p_gridx, p_gridy, V_d[:,0].reshape([Npoint+1,Npoint+1]), cmap='Reds')
			axes[2].plot_surface(p_gridx, p_gridy, V_t[:,1].reshape([Npoint+1,Npoint+1]), cmap='Blues')
			axes[3].plot_surface(p_gridx, p_gridy, V_d[:,1].reshape([Npoint+1,Npoint+1]), cmap='Reds')
			fig.savefig(savepath+'/condstd.pdf')
			# covariance
			fig, axes = plt.subplots(ncols=2, figsize=(12, 4), constrained_layout=True, subplot_kw={"projection": "3d"})
			self.whitebg(axes)
			axes[0].plot_surface(p_gridx, p_gridy, C_t.reshape([Npoint+1,Npoint+1]), cmap='Blues')
			axes[1].plot_surface(p_gridx, p_gridy, C_d.reshape([Npoint+1,Npoint+1]), cmap='Reds')
			fig.savefig(savepath+'/condcostd.pdf')
			plt.close()


	def cond_dis_Enspre(self,model,epoch,savepath):
		## change mannually
		if self.eqn_config.dim==1:
			if self.eqn_config.eqn_name=='Geometric Brownian Motion':
				# condpdf_plotting_points = [0.5,1.0,2.0]
				# int_long = [1.0,1.0,1.0]
				# xlimlen = [1.0,1.0,1.0]
				condpdf_plotting_points = [4.0,5.0,6.0]
				int_long = [3.0,3.0,3.0]
				xlimlen = [4.0,5.0,6.0]
			elif self.eqn_config.eqn_name=='OU Process':
				condpdf_plotting_points = [0.8,1.2,1.8]
				int_long = [0.3,0.3,0.3]
				xlimlen = [0.3,0.3,0.3]
			elif self.eqn_config.eqn_name=='Exp_diffusion':
				condpdf_plotting_points = [-0.3,0.0,0.3]
				int_long = [0.4,0.4,0.4]
				xlimlen = [0.4,0.4,0.4]
			elif self.eqn_config.eqn_name=='Trig_drift':
				condpdf_plotting_points = [0.4,0.5,0.6]
				int_long = [0.4,0.4,0.4]
				xlimlen = [0.4,0.4,0.4]
			elif self.eqn_config.eqn_name=='Exp_OU':
				condpdf_plotting_points = [0.4,0.7,1.0]
				int_long = [0.2,0.2,0.2]
				xlimlen  = [0.1,0.2,0.25]
			elif self.eqn_config.eqn_name=='Double_well':
				condpdf_plotting_points = [-1.5,0,1.5]
				# int_long = [1.0,1.0,1.0]
				# xlimlen  = [1.0,1.0,1.0]
				int_long = [0.5,0.5,0.5]
				xlimlen  = [0.5,0.5,0.5]
			elif self.eqn_config.eqn_name=='Exp_dis':
				condpdf_plotting_points = [0.34,0.52,0.72]
				int_long = [0.15,0.15,0.15]
				xlimlen  = [0.12,0.12,0.12]
			## draw
			font2 = {'size'   : 14,}
			for i in range(len(condpdf_plotting_points)):
				fig1,ax1 = plt.subplots(figsize=(6,4))
				self.condpdf_plotting_std(self.eqn_config.eqn_name,ax1,int_long[i],condpdf_plotting_points[i],self.Delta)
				self.condpdf_plotting_data(self.eqn_config.eqn_name,ax1,model,int_long[i],condpdf_plotting_points[i],self.Delta,N=500000)
				ax1.set_xlabel('x', font2)
				ax1.set_ylabel('pdf', font2)
				ax1.legend(prop=font2)
				if self.eqn_config.eqn_name=='Exp_dis':
					ax1.set_xlim([condpdf_plotting_points[i]-xlimlen[i]/4,condpdf_plotting_points[i]+xlimlen[i]*3/4])
				else:
					ax1.set_xlim([condpdf_plotting_points[i]-xlimlen[i]/2,condpdf_plotting_points[i]+xlimlen[i]/2])
				fig1.savefig(savepath+'/condpdf'+str(i)+'.pdf', bbox_inches='tight')
		elif self.eqn_config.dim==2:
			if self.eqn_config.eqn_name=='MdOU':
				condpdf_plotting_points = [[-2,-1],[0,0],[2,1]]
				int_long = [[0.5,0.5],[0.5,0.5],[0.5,0.5]]
				xlimlen = [[0.5,0.5],[0.5,0.5],[0.5,0.5]]
			elif self.eqn_config.eqn_name=='SO':
				condpdf_plotting_points = [[-0.5,-0.5],[-0.5,0.5],[0.5,0.5]]
				int_long = [[0.5,0.5],[0.5,0.5],[0.5,0.5]]
				xlimlen = [[0.5,0.5],[0.5,0.5],[0.5,0.5]]
			## draw
			font2 = {'size'   : 14,}
			level = [0,3,6,9,12,15,18,21,24]
			if self.eqn_config.eqn_name=='MdOU':
				# for i in range(len(condpdf_plotting_points)):
				# 	fig1,ax1 = plt.subplots(ncols=2, figsize=(12,4))
				# 	self.condpdf_plotting_std2D(self.eqn_config.eqn_name,ax1[0],int_long[i],condpdf_plotting_points[i],self.Delta,level)
				# 	self.condpdf_plotting_data2D(self.eqn_config.eqn_name,ax1[1],model,int_long[i],condpdf_plotting_points[i],self.Delta,level,N=500000)
				# 	fig1.savefig(savepath+'/cond2dpdf'+str(i)+'.pdf', bbox_inches='tight')
				for i in range(len(condpdf_plotting_points)):
					cond2dpdfe = self.condpdf_combine_plotting_std2D(self.eqn_config.eqn_name,int_long[i],condpdf_plotting_points[i],self.Delta,12)
					cond2dpdfe.savefig(savepath+'/cond2dpdfe'+str(i)+'.pdf', bbox_inches='tight')
					cond2dpdfd = self.condpdf_combine_plotting_data2D(self.eqn_config.eqn_name,model,int_long[i],condpdf_plotting_points[i],self.Delta,12,N=500000)
					cond2dpdfd.savefig(savepath+'/cond2dpdfd'+str(i)+'.pdf', bbox_inches='tight')
			for i in range(len(condpdf_plotting_points)):
				fig1,ax1 = plt.subplots(ncols=2, figsize=(12,4))
				self.condmargpdf_plotting_std2D(self.eqn_config.eqn_name,ax1[0],ax1[1],int_long[i],condpdf_plotting_points[i],self.Delta)
				self.condmargpdf_plotting_data2D(self.eqn_config.eqn_name,ax1[0],ax1[1],model,int_long[i],condpdf_plotting_points[i],self.Delta,N=500000)
				ax1[0].set_xlabel('x', font2)
				ax1[0].set_ylabel('pdf', font2)
				ax1[0].legend(prop=font2)
				ax1[1].set_xlabel('x', font2)
				ax1[1].set_ylabel('pdf', font2)
				ax1[1].legend(prop=font2)
				fig1.savefig(savepath+'/condmarginpdf'+str(i)+'.pdf', bbox_inches='tight')

	def Eva_Enspre(self,model,epoch,savepath,DatVes,test_data=None):
		if test_data==None:
			test_data = DatVes.test_data
		else:
			test_data = (sio.loadmat(test_data))['data']
		# # of plot samples
		if self.eqn_config.eqn_name=='Geometric Brownian Motion':
			Num = 20
		elif self.eqn_config.eqn_name=='OU Process':
			Num = 10
		elif self.eqn_config.eqn_name=='Exp_diffusion':
			Num = 5
		elif self.eqn_config.eqn_name=='Trig_drift':
			Num = 1
		elif self.eqn_config.eqn_name=='Exp_OU':
			Num = 10
		elif self.eqn_config.eqn_name=='Double_well':
			Num = 2
		elif self.eqn_config.eqn_name=='Exp_dis':
			Num = 10
		elif self.eqn_config.eqn_name=='MdOU':
			Num = 10
		elif self.eqn_config.eqn_name=='SO':
			Num = 10
		N_T = (test_data).shape[1]
		N_pred = (test_data).shape[-1]
		data_ = DatVes.datachoose((np.vstack(test_data)).T, DatVes.dim, np.zeros([test_data.shape[-1],1],dtype=int), 1)
		Xs = data_[:,:DatVes.dim]
		pre = [Xs]
		for i in range(N_T-1):
			Xs = self.Mulmodel_Generate(model,Xs)
			pre += [Xs]
		pre = np.concatenate(pre, -1)
		pre_ = np.zeros([DatVes.dim,N_T,N_pred])
		for j in range(self.eqn_config.dim):
			pre_[j] = (pre[:,j::DatVes.dim]).T
		for i in range(min(DatVes.dim,10)):
			fig,ax = self.plot_meanstd(test_data[i].T,pre_[i].T,self.eqn_config.dim,self.eqn_config.Delta,savepath=savepath+'/MV'+str(i)+'.pdf')
			self.plotmultipledata((pre_[i].T),[0,(N_T-1)*self.Delta],Num,"OrRd",save=savepath+'/Sample'+str(i)+'.pdf')
		# Debug-only local save disabled in the public package.
		# try:
		# 	sio.savemat('debug_prediction.mat',{'pred':pre_[i].T})
		# except:
		# 	pass
		if self.eqn_config.dim==2:
			if self.eqn_config.eqn_name=='MdOU':
				Nump = 1
			elif self.eqn_config.eqn_name=='SO':
				Nump = 10
			fig1,ax1 = self.plotmultipledata2Dphase(pre_[0].T,pre_[1].T,Nump,"OrRd")
			fig1.savefig(savepath+'/Samplephplot.pdf', bbox_inches='tight')
		# draw the end pdf
		if self.eqn_config.eqn_name=='Double_well':
			self.Compare_end_pdf(test_data[i].T,pre_[i].T,[-1,8000,5000,3000,1000,500,200,50,100],[-5,5],savepath=savepath+'/SDEpdf'+str(i))
		# draw the comparasion of ensemble
		if self.eqn_config.eqn_name=='Trig_drift':
			self.Compare_ensemble1d(model,test_data[i].T,pre_[i].T,self.eqn_config.Delta,savepath=savepath+'/ensem')

	def fakemeanvar_Enspre(self,model,data,epoch,savepath):
		# data = tf.cast(data,tf.float32)
		X_sample = self.Ens_fake_sample(model,data)
		## fake mean/variance test
		for i in range(self.eqn_config.dim):
			X_sample_t = X_sample[:,i::self.eqn_config.dim]
			data_t = data[:,i::self.eqn_config.dim]
			p_grid = np.arange(X_sample_t.shape[1])+1
			Mean_t, Mean_d = np.mean(data_t,axis=0),np.mean(X_sample_t,axis=0)
			Ecov_t, Ecov_d = -np.sort(-np.linalg.eig(np.cov((data_t).T))[0]),-np.sort(-np.linalg.eig(np.cov(X_sample_t.T))[0])
			# draw
			font2 = {'size'   : 14,}
			fig1,ax1 = plt.subplots(figsize=(6,4))
			ax1.plot(p_grid,Mean_t,linestyle='-',color='#000080',marker='o',markerfacecolor="None", label='Ground Truth')
			ax1.plot(p_grid,Mean_d,linestyle='dashed',color='#DC143C',marker='s',markerfacecolor="None", label='Prediction')
			ax1.set_xlabel('Component',font2)
			ax1.set_ylabel('Mean',font2)
			ax1.legend(prop=font2)
			fig1.savefig(savepath+'/fakemean'+str(i)+'.pdf', bbox_inches='tight')

			fig2,ax2 = plt.subplots(figsize=(6,4))
			ax2.plot(p_grid,Ecov_t,linestyle='-',color='#000080',marker='o',markerfacecolor="None", label='Ground Truth')
			ax2.plot(p_grid,Ecov_d,linestyle='dashed',color='#DC143C',marker='s',markerfacecolor="None", label='Prediction')
			ax2.set_yscale('log')
			ax2.set_xlabel('Component',font2)
			ax2.set_ylabel('Spectra of Cov Matrix',font2)
			ax2.legend(prop=font2)
			fig2.savefig(savepath+'/fakevar'+str(i)+'.pdf', bbox_inches='tight')
		
		## sample test
		condpdf_plotting_points = [4,10,30]
		int_long = [0.3,0.3,0.3]
		for i in range(len(condpdf_plotting_points)):
			fig1,ax1 = plt.subplots(figsize=(6,4))
			self.pdf_plotting_data(ax1,data[:,condpdf_plotting_points[i]-1],int_long[i],'-','#000080',label_='Ground Truth')
			# ax1.hist(X_sample[:,condpdf_plotting_points[i]-1], bins=50, alpha=0.6, ec="k", density=True, histtype='stepfilled',label='Learned')
			self.pdf_plotting_data(ax1,X_sample[:,condpdf_plotting_points[i]-1],int_long[i],'dashed','#DC143C',label_='Prediction')
			ax1.set_xlabel('x', font2)
			ax1.set_ylabel('pdf', font2)
			ax1.legend(prop=font2)
			fig1.savefig(savepath+'/fakepdf'+str(i)+'.pdf', bbox_inches='tight')

	def plot_sampletraindata(self,DatVes,savepath):
		if self.eqn_config.eqn_name=='Geometric Brownian Motion':
			dataxlim,trainlim,Num = [[0,1]],[1],20
		elif self.eqn_config.eqn_name=='OU Process':
			dataxlim,trainlim,Num = [[0,4]],[1],10
		elif self.eqn_config.eqn_name=='Exp_diffusion':
			dataxlim,trainlim,Num = [[0,10]],[1],5
		elif self.eqn_config.eqn_name=='Trig_drift':
			dataxlim,trainlim,Num = [[0,10]],[1],1
		elif self.eqn_config.eqn_name=='Exp_OU':
			dataxlim,trainlim,Num = [[0,5]],[1],10
		elif self.eqn_config.eqn_name=='Double_well':
			dataxlim,trainlim,Num = [[0,1]],[1],10
		elif self.eqn_config.eqn_name=='Exp_dis':
			dataxlim,trainlim,Num = [[0,5]],[1],10
		elif self.eqn_config.eqn_name=='MdOU':
			dataxlim,trainlim,Num = [[0,5],[0,5]],[1,1],10
		elif self.eqn_config.eqn_name=='SO':
			dataxlim,trainlim,Num = [[0,5],[0,5]],[1,1],10
		dataset = sio.loadmat(DatVes.dat_config.TrainData_dir)
		for i in range(self.eqn_config.dim):
			data = dataset['data'][i].T
			l1,l2 = np.max(data[:Num]),np.min(data[:Num])
			fig1,ax1 = self.plotmultipledata(data,[0,1],Num,"PuBu")
			ax1.plot(trainlim[i]*np.ones(10),np.linspace(l2-abs(l2)-5,l1+abs(l1),10),color='gray',linestyle='dashed',linewidth=1.0,alpha=0.5)
			ax1.set_xlim([dataxlim[i][0]-0.05*(dataxlim[i][1]-dataxlim[i][0]),dataxlim[i][1]+0.05*(dataxlim[i][1]-dataxlim[i][0])])
			ax1.set_ylim([l2-0.05*(l1-l2),l1+0.05*(l1-l2)])
			fig1.savefig(savepath+'/traindata'+str(i)+'.pdf', bbox_inches='tight')
		if self.eqn_config.dim==2:
			if self.eqn_config.eqn_name=='MdOU':
				Nump = 1
			elif self.eqn_config.eqn_name=='SO':
				Nump = 20
			fig1,ax1 = self.plotmultipledata2Dphase(dataset['data'][0].T,dataset['data'][1].T,Nump,"PuBu")
			fig1.savefig(savepath+'/traindataphplot.pdf', bbox_inches='tight')

	#---------------------------------tools----------------------------------------------

	def plot_meanstd(self,testdata,predictdata,Delta,Resdata=None,slice=0,savepath=None):
		# data should be in the form of Ndata*test
		# Test data
		xt_test = np.arange(testdata.shape[-1])*Delta
		xmean_test = np.mean(testdata,axis=0)
		xstde_test = np.std(testdata,axis=0,ddof=1)
		xt_test,xmean_test,xstde_test = xt_test[slice:],xmean_test[slice:],xstde_test[slice:]
		# Predict data
		xt_pred = np.arange(predictdata.shape[-1])*Delta
		xmean_pred = np.mean(predictdata,axis=0)
		xstde_pred = np.std(predictdata,axis=0,ddof=1)
		xt_pred,xmean_pred,xstde_pred = xt_pred[slice:],xmean_pred[slice:],xstde_pred[slice:]
		# Resdata
		# if Resdata is not None:
		# 	xmean_pred = xmean_pred+Resdata[:xmean_pred.shape[0]]
		# Bound
		test_l,test_u = xmean_test - xstde_test, xmean_test + xstde_test
		pred_l,pred_u = xmean_pred - xstde_pred, xmean_pred + xstde_pred
		print('Mean Error: '+"{:.4E}".format(abs(xmean_test[-1]-xmean_pred[-1])))
		print('Mean True: '+"{:.4E}".format(xmean_test[-1]))
		print('Std Error: '+"{:.4E}".format(abs(xstde_test[-1]-xstde_pred[-1])))
		print('Std True: '+"{:.4E}".format(xstde_test[-1]))
		# plot
		font2 = {'size'   : 14,}
		font3 = {'size'   : 8.5,}
		fig1, ax1 = plt.subplots(figsize=[6,4])
		ax1.plot(xt_test, xmean_test, linewidth=2.0, color='#000080', label='Ground Truth Mean')
		ax1.fill_between(xt_test, test_l, test_u, color='#000080', alpha=0.2, label='Ground Truth Std')
		ax1.plot(xt_pred, xmean_pred, linewidth=2.0, color='#DC143C', linestyle='dashed', label='Prediction Mean')
		ax1.fill_between(xt_pred, pred_l, pred_u, color='#DC143C', alpha=0.2, label='Prediction Std')
		ylow,yup = min(np.min(test_l),np.min(pred_l)),max(np.max(test_u),np.max(pred_u))
		# ax1.set_ylim([ylow-0.03*abs(ylow),yup+0.03*abs(yup)])
		# ax1.set_ylim([ylow-0.3*abs(ylow),yup+0.3*abs(yup)])
		ypad = 0.05*(yup-ylow) if yup>ylow else 0.05*abs(yup)
		ax1.set_ylim([ylow-ypad,yup+ypad])
		ax1.set_xlabel('T', font2)
		# ax1.set_ylabel('pdf', font2)
		# ax1.legend(prop=font3, ncol=2)
		ax1.legend(prop=font3, ncol=2, loc="lower right")
		if savepath is not None:
			fig1.savefig(savepath,bbox_inches='tight')
		return fig1,ax1

	def plotmultipledata(self,dataset,Tinterval,num,cmapname,save=False):
		Nx = dataset.shape[-1]
		x = np.linspace(Tinterval[0],Tinterval[1],Nx)
		cmap = plt.get_cmap(cmapname)
		if num>1:
			colors = [cmap(i) for i in np.linspace(0.3, 1, num)]
		elif num==1:
			colors = [cmap(1.0)]
		fig1,ax1 = plt.subplots(figsize=(6,4))
		font2 = {'size'   : 14,}
		for i, color in enumerate(colors, start=0):
			ax1.plot(x, dataset[i], color=color)
		ax1.set_xlabel('T', font2)
		if save:
			fig1.savefig(save, bbox_inches='tight')
		return fig1,ax1

	# def plotmultipledata(dataset,Tinterval,num,cmapname,save=False):
	# 	# for double well
	# 	Nx = dataset.shape[-1]
	# 	x = np.linspace(Tinterval[0],Tinterval[1],Nx)
	# 	oe = np.ones(x.shape)
	# 	ne = -oe
	# 	cmap = plt.get_cmap(cmapname)
	# 	if num>1:
	# 		colors = [cmap(i) for i in np.linspace(0.3, 1, num)]
	# 	elif num==1:
	# 		colors = [cmap(0.8)]
	# 	fig1,ax1 = plt.subplots(figsize=(12,4))
	# 	ax1.plot(x, oe, color='grey', alpha=0.8, linewidth=0.5)
	# 	ax1.plot(x, ne, color='grey', alpha=0.8, linewidth=0.5)
	# 	font2 = {'size'   : 14,}
	# 	for i, color in enumerate(colors, start=0):
	# 		ax1.plot(x, dataset[i], color=color)
	# 	ax1.set_xlabel('T', font2)
	# 	plt.xlim([0,500])
	# 	if save:
	# 		fig1.savefig(save, bbox_inches='tight')
	# 	return fig1,ax1

	def plotmultipledata2Dphase(self,dataset1,dataset2,num,cmapname,save=False):
		cmap = plt.get_cmap(cmapname)
		if num>1:
			colors = [cmap(i) for i in np.linspace(0.3, 1, num)]
		elif num==1:
			colors = [cmap(1.0)]
		fig1,ax1 = plt.subplots(figsize=(6,4))
		font2 = {'size'   : 14,}
		for i, color in enumerate(colors, start=0):
			ax1.plot(dataset1[i], dataset2[i], color=color)
		ax1.set_xlabel('$x_1$', font2)
		ax1.set_ylabel('$x_2$', font2)
		if save:
			fig1.savefig(save, bbox_inches='tight')
		return fig1,ax1

	def Ens_fake_sample(self,model,data):
		Xs = data[:,:model[0].dim]
		X_sample = [Xs]
		for i in range(model[0].d_RNN-1):
			Xincre = self.Mulmodel_Generate_pure(model,Xs)
			Xs = Xs+Xincre
			X_sample += [Xincre]
		X_sample = np.concatenate(X_sample, -1)
		return X_sample

	def Mulmodel_Generate_pure(self,modellist,Xs):
		Nmodel = len(modellist)
		modelid = np.random.randint(Nmodel, size=Xs.shape[0])
		Xre = np.zeros(Xs.shape)
		for j in range(Nmodel):
			_id = np.where(modelid==j)[0]
			Z_samp = tf.random.normal([_id.shape[0], modellist[0].n_Z])
			Xre[_id] = modellist[j].Generate(Xs[_id],Z_samp)
		return Xre

	def Singlemodel_Generate(self,model,Xs,N_T):
		X_sample = [Xs]
		for i in range(N_T-1):
			Z_samp = tf.random.normal([Xs.shape[0], model.n_Z])
			Xincre = model.Generate(Xs,Z_samp)
			Xs = Xs+Xincre
			X_sample += [Xs]
		X_sample = np.concatenate(X_sample, -1)
		return X_sample

	def Compare_end_pdf(self,testdata,predata,indexes,x_range,savepath):
		x_axis = np.linspace(x_range[0],x_range[1],500)
		for index in indexes:
			tdata1,pdata1 = testdata[:,index],predata[:,index]
			fig1,ax1 = plt.subplots(figsize=(6,4))
			font2 = {'size'   : 14,}
			kdet = scipy.stats.kde.gaussian_kde(tdata1)
			kdep = scipy.stats.kde.gaussian_kde(pdata1)
			ax1.plot(x_axis, kdet(x_axis), color='#000080',label='Reference')
			ax1.plot(x_axis, kdep(x_axis), color='#DC143C',linestyle='dashed',label='Learned')
			ax1.legend(prop=font2)
			fig1.savefig(savepath+'_'+str(index)+'.pdf', bbox_inches='tight')

	def Compare_ensemble1d(self,model,testdata,predata,Delta,savepath):
		Nmodel = len(model)
		gap = self.gap_withlast(testdata.shape[-1],35)
		figm,axm = plt.subplots(figsize=(6,4))
		figv,axv = plt.subplots(figsize=(6,4))
		font2 = {'size'   : 14,}
		font3 = {'size'   : 11,}
		N_T,N_pred = testdata.shape[1],testdata.shape[0]
		Xs = testdata[:,:model[0].dim]
		pre_av = np.zeros([N_pred,N_T])
		xt_test = np.arange(testdata.shape[-1])*Delta
		p = int(N_pred/Nmodel)
		ite = range(Nmodel)
		# modification
		ite = [0,3,4]
		colors = ['#8E0202','#D00000','#FF0000']
		for i in range(len(ite)):
			testdatasingle = self.Singlemodel_Generate(model[ite[i]],Xs,N_T)
			xmean_pre = np.mean(testdatasingle,axis=0)
			xstde_pre = np.std(testdatasingle,axis=0,ddof=1)
			# axm.plot(xt_test, xmean_pre, linewidth=2.0, color='#DC143C', linestyle='dashed', alpha=0.7)
			# axv.plot(xt_test, xstde_pre, linewidth=2.0, color='#DC143C', linestyle='dashed', alpha=0.7)
			axm.plot(xt_test, xmean_pre, linewidth=2.0, color=colors[i], linestyle='dashed', label='Unaverage Model '+str(i+1), alpha=0.7)
			axv.plot(xt_test, xstde_pre, linewidth=2.0, color=colors[i], linestyle='dashed', label='Unaverage Model '+str(i+1), alpha=0.7)
			pre_av[ite[i]*p:(ite[i]+1)*p] = testdatasingle[:p]
		# axm.plot([], [], linewidth=2.0, color='#DC143C', linestyle='dashed', label='Unaverage Models', alpha=0.7)
		# axv.plot([], [], linewidth=2.0, color='#DC143C', linestyle='dashed', label='Unaverage Models', alpha=0.7)
		# ground truth
		xmean_test = np.mean(testdata,axis=0)
		xstde_test = np.std(testdata,axis=0,ddof=1)
		# axm.plot(xt_test[gap], xmean_test[gap], linewidth=2.0, color='black',marker='s',markerfacecolor="None", label='Ground Truth')
		# axv.plot(xt_test[gap], xstde_test[gap], linewidth=2.0, color='black',marker='s',markerfacecolor="None", label='Ground Truth')
		axm.plot(xt_test[gap], xmean_test[gap], linewidth=2.0, color='black',label='Ground Truth')
		axv.plot(xt_test[gap], xstde_test[gap], linewidth=2.0, color='black',label='Ground Truth')
		# improper ensemble
		# xmean_pre = np.mean(pre_av,axis=0)
		# xstde_pre = np.std(pre_av,axis=0,ddof=1)
		# axm.plot(xt_test[gap], xmean_pre[gap], linewidth=2.0, color='#008B8B',marker='D',markerfacecolor="None", linestyle='dashdot', label='Improper Ensemble')
		# axv.plot(xt_test[gap], xstde_pre[gap], linewidth=2.0, color='#008B8B',marker='D',markerfacecolor="None", linestyle='dashdot', label='Improper Ensemble')
		# probabilistic enseble
		xmean_pre = np.mean(predata,axis=0)
		xstde_pre = np.std(predata,axis=0,ddof=1)
		axm.plot(xt_test[gap], xmean_pre[gap], linewidth=2.0, color='#66B2FF',marker='o',markerfacecolor="None", linestyle='dotted', label='Prob Ensemble')
		axv.plot(xt_test[gap], xstde_pre[gap], linewidth=2.0, color='#66B2FF',marker='o',markerfacecolor="None", linestyle='dotted', label='Prob Ensemble')
		# label
		axm.set_xlabel('T', font2)
		axm.set_ylabel('Mean', font2)
		axm.legend(prop=font3)
		axv.set_xlabel('T', font2)
		axv.set_ylabel('Std', font2)
		axv.legend(prop=font3)
		# save
		figm.savefig(savepath+'m.pdf', bbox_inches='tight')
		figv.savefig(savepath+'v.pdf', bbox_inches='tight')

	def gap_withlast(self,length,N):
		a = np.arange(length)
		b = a[::N]
		if b[-1]!=a[-1]:
			np.append(b,a[-1])
		return b

	def whitebg(self,axes):
		for ax in axes:
			ax.xaxis.pane.fill = False
			ax.yaxis.pane.fill = False
			ax.zaxis.pane.fill = False
			ax.xaxis.pane.set_edgecolor('w')
			ax.yaxis.pane.set_edgecolor('w')
			ax.zaxis.pane.set_edgecolor('w')


class DataTran():
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
		self.N_pred          = self.dat_config.N_pred
		self.Monitor         = Monitor

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

	# def test_mdat1model(self,model,save_path,mode='w'):
	# 	L_Nmax_Test = (self.test_data).shape[1]
	# 	self.pred = np.zeros([self.dim,L_Nmax_Test,self.N_pred])
	# 	for i in range(self.N_pred):
	# 		data_ = (self.test_data[:,:,i]).flatten()
	# 		pred  = self.test_singledata(data_,model,L_Nmax_Test)
	# 		self.pred[:,:,i] = pred
	# 	if mode=='w':
	# 		sio.savemat(save_path,{'pred':self.pred})
	# 	elif mode=='a':
	# 		if os.path.exists(save_path):
	# 			data_exist = (sio.loadmat(save_path))['pred']
	# 			self.pred = np.concatenate([data_exist,self.pred],axis=-1)
	# 		sio.savemat(save_path,{'pred':self.pred})

	def test_mdat1model(self,model,save_path,mode='w'):
		L_Nmax_Test = (self.test_data).shape[1]
		Nullstart = True if ('Resdata' in self.eqn_config.keys()) and self.eqn_config.Resdata['if'] else False
		self.pred = self.test_tensordata(self.test_data,model,L_Nmax_Test,Nullstart)
		if mode=='w':
			sio.savemat(save_path,{'pred':self.pred})
		elif mode=='a':
			if os.path.exists(save_path):
				data_exist = (sio.loadmat(save_path))['pred']
				self.pred = np.concatenate([data_exist,self.pred],axis=-1)
			sio.savemat(save_path,{'pred':self.pred})

	def train_data_trans(self,seed_):
		smaple_L_Nmax = self.L_Nmax-self.d_RNN
		# random setting
		np.random.seed(seed_)
		sample_init_L = np.random.randint(smaple_L_Nmax+1,size=(self.N_long_traj,self.n_ea_traj))
		temp_wu = np.random.permutation(self.n_train)
		if ('zeroinit' in self.dat_config.keys()) and self.dat_config.zeroinit:
			sample_init_L = np.zeros(sample_init_L.shape,dtype=int)
		# data merging
		data_ = (np.vstack(self.train_data)).T
		# set train inputs and outputs
		train_mat = np.zeros((self.n_train, self.dim*self.d_RNN))
		for i in range(self.n_ea_traj):
			train_mat[i*self.N_long_traj:(i+1)*self.N_long_traj] = self.datachoose(data_, self.dim, sample_init_L[:,[i]], self.d_RNN)
		self.train_mat  = train_mat[temp_wu,:]
		# monitor
		if self.Monitor.monitor_config.traindata_hist:
			# take the first variable
			for i in range(self.dim):
				self.Monitor.data2dhistogram(self.train_mat[:,i:self.dim*self.d_RNN:self.dim],self.eqn_config.Delta,"Train_data"+str(i))
		if self.Monitor.monitor_config.traintransin_hist:
			# take the first variable
			for i in range(self.dim):
				self.Monitor.transprobinfo(self.train_mat[:,i:self.dim*self.d_RNN:self.dim],"Train_data"+str(i))

	def test_singledata(self,test_data,model,N_T):
		# data is in the form of [dim*n_of_time_step]
		# aranging as [dim1_tracj, dim2_tracj,...]
		data_ = self.datachoose(test_data, self.dim, 0, 1)
		pre = np.zeros(N_T*self.dim)
		pre[:self.dim] = data_
		for i in range(N_T-1):
			next_time = model.predict(np.array([pre[self.dim*i:self.dim*(i+1)]]))
			pre[self.dim*(i+1):self.dim*(i+2)] = next_time
		pre = (pre.reshape([N_T,self.dim])).T
		return pre

	def test_tensordata(self,test_data,model,N_T,Nullstart=False):
		# data is in the form of [dim*n_of_time_step]
		# aranging as [dim1_tracj, dim2_tracj,...]
		data_ = self.datachoose((np.vstack(test_data)).T, self.dim, np.zeros([test_data.shape[-1],1],dtype=int), 1)
		Xs = data_[:,:self.dim]
		if Nullstart:
			Xs = np.zeros(Xs.shape)
		pre = [Xs] 
		for i in range(N_T-1):
			Z_samp = tf.random.normal([data_.shape[0], model.n_Z])
			Xs = model.Generate(Xs,Z_samp)
			pre += [Xs]
		pre = np.concatenate(pre, -1)
		pre_ = np.zeros([self.dim,N_T,self.N_pred])
		for j in range(self.dim):
			pre_[j] = (pre[:,j::self.dim]).T
		return pre_

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
