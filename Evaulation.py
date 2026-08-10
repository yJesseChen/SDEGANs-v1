from __future__ import division
import pdb
import os
import munch
import json
import logging

import numpy as np
import numpy.linalg
import matplotlib
from matplotlib import pyplot as plt
import scipy
import scipy.io as sio
from scipy import stats
import tensorflow as tf

class Evaluate():
	def plot_ode_compare(self,testdata,predictdata,Delta,savepath=None):
		xt = np.arange(testdata.shape[0])*Delta
		xp = np.arange(predictdata.shape[0])*Delta
		T = max(xt[-1],xp[-1])
		fig1, ax1 = plt.subplots(figsize=[10,7])
		ax1.plot(xt, testdata, color='black')
		ax1.plot(xp, predictdata, 'o', color='#6495ED', markerfacecolor='none')
		plt.xlim([-0.1*T,1.1*T])
		if savepath is not None:
			fig1.savefig(savepath)
		return fig1,ax1

	def plot_train_hisGAN(self,Nepoc,G_loss,D_loss,savepath=None):
		x = np.arange(len(G_loss))
		fig1, ax1 = plt.subplots(figsize=[10,7])
		# ax1.plot(x, G_loss, color='#4169E1', label='Generator')
		# ax1.plot(x, D_loss, color='#DC143C', label='Discriminator')
		gp = ax1.plot(x, G_loss, color='#4169E1', label='Generator')
		ax1.tick_params(axis='y', labelcolor='#4169E1')
		ax2 = ax1.twinx()
		dp = ax2.plot(x, -np.array(D_loss), color='#DC143C', label='Negative Discriminator')
		ax2.tick_params(axis='y', labelcolor='#DC143C')
		ax2.set_yscale('log')
		ax1.set_xlim([-100,Nepoc+100])
		# fig1.tight_layout()
		gdps = gp+dp
		labs = [l.get_label() for l in gdps]
		ax1.legend(gdps,labs)
		if savepath is not None:
			fig1.savefig(savepath)
		return fig1,ax1

	def plot_index(self,Nepoc,data,name,savepath=None,log=True):
		x = np.arange(Nepoc)
		fig1, ax1 = plt.subplots(figsize=[10,7])
		ax1.plot(x, data, color='#0000FF', label=name)
		if log:
			ax1.set_yscale('log')
		ax1.legend()
		if savepath is not None:
			fig1.savefig(savepath)
		return fig1,ax1

	def plot_sample(self,testdata,predictdata,Delta,slice=0,savepath=None):
		# data should be in the form of Ndata*test
		# Test data
		xt_test = np.arange(testdata.shape[-1])*Delta
		# Predict data
		xt_pred = np.arange(predictdata.shape[-1])*Delta
		# plot
		fig1, ax1 = plt.subplots(1,2,figsize=[20,7])
		for i in range(min(testdata.shape[0],200)):
			ax1[0].plot(xt_test, testdata[i])
			ax1[1].plot(xt_pred, predictdata[i])
		ax1[0].set_title('Ground Truth')
		ax1[1].set_title('Prediction')
		if savepath is not None:
			fig1.savefig(savepath)
		return fig1,ax1

	# def plot_meanstd(self,testdata,predictdata,Delta,Resdata=None,slice=0,savepath=None):
	# 	# data should be in the form of Ndata*test
	# 	# Test data
	# 	xt_test = np.arange(testdata.shape[-1])*Delta
	# 	xmean_test = np.mean(testdata,axis=0)
	# 	xstde_test = np.std(testdata,axis=0,ddof=1)
	# 	xt_test,xmean_test,xstde_test = xt_test[slice:],xmean_test[slice:],xstde_test[slice:]
	# 	# Predict data
	# 	xt_pred = np.arange(predictdata.shape[-1])*Delta
	# 	xmean_pred = np.mean(predictdata,axis=0)
	# 	xstde_pred = np.std(predictdata,axis=0,ddof=1)
	# 	xt_pred,xmean_pred,xstde_pred = xt_pred[slice:],xmean_pred[slice:],xstde_pred[slice:]
	# 	# Resdata
	# 	if Resdata is not None:
	# 		xmean_pred = xmean_pred+Resdata[:xmean_pred.shape[0]]
	# 	# Bound
	# 	test_l,test_u = xmean_test - xstde_test, xmean_test + xstde_test
	# 	pred_l,pred_u = xmean_pred - xstde_pred, xmean_pred + xstde_pred
	# 	# plot
	# 	fig1, ax1 = plt.subplots(figsize=[10,7])
	# 	ax1.plot(xt_test, xmean_test, color='#4169E1', label='Ground Truth')
	# 	ax1.fill_between(xt_test, test_l, test_u, color='#4169E1', alpha=0.2)
	# 	ax1.plot(xt_pred, xmean_pred, color='#DC143C', label='Prediction')
	# 	ax1.fill_between(xt_pred, pred_l, pred_u, color='#DC143C', alpha=0.2)
	# 	ax1.set_ylim([min(np.min(test_l),np.min(pred_l)),max(np.max(test_u),np.max(pred_u))])
	# 	# ax1.set_ylim([-1.5,2.5])
	# 	ax1.legend()
	# 	if savepath is not None:
	# 		fig1.savefig(savepath)
	# 	return fig1,ax1

	def plot_meanstd(self,testdataMD,predictdataMD,dim,Delta,Resdata=None,slice=0,savepath=None):
		# data should be in the form of dim*Ndata*test
		# N_plot = min(dim,10)
		# fig1, ax1 = plt.subplots(ncols=N_plot, figsize=(10*N_plot, 7), squeeze=False)
		n_col = 5
		n_row = dim//n_col+int(dim%n_col!=0)
		if dim<=5:
			n_col = dim
		fig1, axes = plt.subplots(nrows=n_row, ncols=n_col, figsize=(n_col*3, n_row*2), constrained_layout=True, squeeze=False)
		for i in range(n_row):
			for j in range(n_col):
				num = i*n_col+j
				if num<=(dim-1):
					# Test data
					testdata,predictdata = testdataMD[num].T,predictdataMD[num].T
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
					if Resdata is not None:
						xmean_pred = xmean_pred+Resdata[:xmean_pred.shape[0]]
					# Bound
					test_l,test_u = xmean_test - xstde_test, xmean_test + xstde_test
					pred_l,pred_u = xmean_pred - xstde_pred, xmean_pred + xstde_pred
					# plot
					axes[i,j].plot(xt_test, xmean_test, color='#4169E1', label='Ground Truth')
					axes[i,j].fill_between(xt_test, test_l, test_u, color='#4169E1', alpha=0.2)
					axes[i,j].plot(xt_pred, xmean_pred, color='#DC143C', label='Prediction')
					axes[i,j].fill_between(xt_pred, pred_l, pred_u, color='#DC143C', alpha=0.2)
					axes[i,j].set_ylim([min(np.min(test_l),np.min(pred_l)),max(np.max(test_u),np.max(pred_u))])
					# axes.set_ylim([-1.5,2.5])
					axes[i,j].legend()
				else:
					break
		if savepath is not None:
			fig1.savefig(savepath)
		return fig1,axes

	def plot_endpdfGeneralD(self,testdataMD,predictdataMD,dim,savepath=None):
		# data should be in the form of dim*Ndata
		# N_plot = min(dim,10)
		# fig1, ax1 = plt.subplots(ncols=N_plot, figsize=(10*N_plot, 7), squeeze=False)
		x_axis = np.linspace(-5,5,200)
		# for i in range(N_plot):
		# 	kde = scipy.stats.kde.gaussian_kde(testdataMD[i])
		# 	ax1[0,i].plot(x_axis, kde(x_axis), color='#4169E1',label='Ground Truth')
		# 	kde = scipy.stats.kde.gaussian_kde(predictdataMD[i])
		# 	ax1[0,i].plot(x_axis, kde(x_axis), color='#DC143C',label='Prediction')
		# 	ax1[0,i].legend()
		n_col = 5
		n_row = dim//n_col+int(dim%n_col!=0)
		if dim<=5:
			n_col = dim
		fig1, axes = plt.subplots(nrows=n_row, ncols=n_col, figsize=(n_col*3, n_row*2), constrained_layout=True, squeeze=False)
		for i in range(n_row):
			for j in range(n_col):
				num = i*n_col+j
				if num<=(dim-1):
					mt,st = np.mean(testdataMD[num]),np.std(testdataMD[num])
					mp,sp = np.mean(predictdataMD[num]),np.std(predictdataMD[num])
					x_axis = np.linspace(min(mt-3*st,mp-3*sp),max(mt+3*st,mp+3*sp),500)
					kde = scipy.stats.kde.gaussian_kde(testdataMD[num])
					axes[i,j].plot(x_axis, kde(x_axis), color='#4169E1',label='Ground Truth')
					kde = scipy.stats.kde.gaussian_kde(predictdataMD[num])
					axes[i,j].plot(x_axis, kde(x_axis), color='#DC143C',label='Prediction')
					axes[i,j].legend()
				else:
					break
		if savepath is not None:
			fig1.savefig(savepath)
		return fig1,axes

	def readmodel(path,Model,config):
		# This function is designed for test for single models
		ModelX = Model(config)
		# see https://www.tensorflow.org/api_docs/python/tf/train/Checkpoint
		ModelXCheckp = tf.train.Checkpoint(G_optimizer=ModelX.G_optimizer,D_optimizer=ModelX.D_optimizer,G=ModelX.G,D=ModelX.D)
		manager = tf.train.CheckpointManager(ModelXCheckp, path, max_to_keep=1)
		ModelXCheckp.restore(manager.latest_checkpoint).expect_partial()
		return ModelX

class SdeGanEva(Evaluate):
	def __init__(self,config,result_path,save_path):
		self.eqn_config  = config.eqn_config
		self.net_config  = config.net_config
		self.dat_config  = config.dat_config
		self.result_path = result_path
		self.save_path   = save_path
		self.dim = self.eqn_config.dim
		self.Delta   = self.eqn_config.Delta
		self.n_epochs = self.net_config.N_epochs
		self.test_data_path  = self.dat_config.TestData_dir
		if not os.path.exists(self.save_path):
			os.makedirs(self.save_path)

	def plot_samplecompare(self,save=False):
		test_data = (sio.loadmat(self.test_data_path))['data']
		try:
			pre_data = (sio.loadmat(self.result_path+'/predict.mat'))['pred']
		except:
			raise AttributeError('ResnetEva::plot_single: Fail to find prediction data')
		for i in range(min(self.dim,10)):
			save_ = (self.save_path+'/S'+str(i+1)+'.pdf') if save else None
			# if i==0:
			# 	pdb.set_trace()
			fig,ax = self.plot_sample(test_data[i].T,pre_data[i].T,self.Delta,savepath=save_)

	def plot_losthist(self,save=False):
		try:
			with open(self.result_path+'/Test_history.json') as json_data_file:
				file = json.load(json_data_file)
				G_loss_data = file['G_loss']
				D_loss_data = file['D_loss']
		except:
			raise AttributeError('SdeGanEva::plot_losshist: Fail to find loss data')
		save_ = (self.save_path+'/loss_hist.pdf') if save else None
		fig,ax = self.plot_train_hisGAN(self.n_epochs,G_loss_data,D_loss_data,savepath=save_)

	def plot_Wdistance(self,save=False):
		try:
			with open(self.result_path+'/Test_history.json') as json_data_file:
				file = json.load(json_data_file)
				W_dist_data = file['W_dist']
				save_ = (self.save_path+'/W_dist.pdf') if save else None
				fig,ax = self.plot_index(self.n_epochs,W_dist_data,'Wasserstein Distance',savepath=save_,log=False)
		except:
			pass

	def plot_meancompare(self,save=False,epoch=''):
		test_data = (sio.loadmat(self.test_data_path))['data']
		try:
			pre_data = (sio.loadmat(self.result_path+'/predict.mat'))['pred']
		except:
			raise AttributeError('ResnetEva::plot_single: Fail to find prediction data')
		# for i in range(min(self.dim,10)):
		# 	save_ = (self.save_path+'/'+epoch+'M'+str(i+1)+'.pdf') if save else None
		# 	# pdb.set_trace()
		# 	fig,ax = self.plot_meanstd(test_data[i].T,pre_data[i].T,self.dim,self.Delta,savepath=save_)
		save_ = (self.save_path+'/'+epoch+'M'+'.pdf') if save else None
		fig,ax = self.plot_meanstd(test_data,pre_data,self.dim,self.Delta,savepath=save_)

	def plot_meancompare_Resplus(self,save=False,epoch=''):
		test_data = (sio.loadmat(self.test_data_path))['data']
		Res_data = (sio.loadmat(self.eqn_config.Resdata['path']))['pred']
		try:
			pre_data = (sio.loadmat(self.result_path+'/predict.mat'))['pred']
		except:
			raise AttributeError('ResnetEva::plot_single: Fail to find prediction data')
		for i in range(min(self.dim,10)):
			save_ = (self.save_path+'/'+epoch+'M'+str(i+1)+'.pdf') if save else None
			fig,ax = self.plot_meanstd(test_data[i].T,pre_data[i].T,self.Delta,Resdata=Res_data[i],savepath=save_)



