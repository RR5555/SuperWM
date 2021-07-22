
from svm import svm_raw_dataset_for_n_back_diff,\
	thunder_svm_raw_dataset_for_n_back_diff,\
	thunder_svm_raw_dataset_for_n_back_diff_agg,\
	pth_SVM
from base_fct import stop_watch,\
	print_N_log
from train_test import run_a_model,\
	test_a_model
	
from seed import init_seeds
from save_N_load import save_obj
from shared_configuration import RESULT_DIR
from dataset import n_back_dataset

import os
import sys
import time
import numpy as np
import torch
from functools import partial
import logging

# SVM on raw initial data for n-back differentiation
# 'HCP2021WM', 'HCP2021WM_BEH'
def svm_raw_dataset_for_n_back_diff_exp(dataset_name, _test_size=.33, _random_state=0, _path=None):
	prog_time_start = time.time()
	SVM, X_train, X_test, y_train, y_test = svm_raw_dataset_for_n_back_diff(dataset_name=dataset_name, _test_size=_test_size, _random_state=_random_state)
	if _path is not None:
		save_obj(SVM.get_params(), _path)
	print(f"Training accuracy: {SVM.score(X_train, y_train)}")
	print(f"Testing accuracy: {SVM.score(X_test, y_test)}")
	stop_watch(prog_time_start)

def score_SVM(_model, _X, _y):
	return np.mean(_model.predict(_X)==_y)

def thunder_svm_raw_dataset_for_n_back_diff_exp_agg(dataset_name, _test_size=.33, _random_state=0, _path=None, _remove_mean=False, _preproc=lambda x:x):
	prog_time_start = time.time()
	SVM, X_train, X_test, y_train, y_test = thunder_svm_raw_dataset_for_n_back_diff_agg(dataset_name=dataset_name, _test_size=_test_size, _random_state=_random_state, _remove_mean=_remove_mean, _preproc=_preproc)
	if _path is not None:
		SVM.save_to_file(_path)
	print(f"Training accuracy: {SVM.score(X_train, y_train)}")
	# print(f"Training accuracy: {score_SVM(SVM, X_train, y_train)}")
	print(f"Testing accuracy: {SVM.score(X_test, y_test)}")
	# print(f"Testing accuracy: {score_SVM(SVM, X_test, y_test)}")
	stop_watch(prog_time_start)


def thunder_svm_raw_dataset_for_n_back_diff_exp(dataset_name, _test_size=.33, _random_state=0, _path=None, _remove_mean=False, _preproc=lambda x:x):
	prog_time_start = time.time()
	SVM, X_train, X_test, y_train, y_test = thunder_svm_raw_dataset_for_n_back_diff(dataset_name=dataset_name, _test_size=_test_size, _random_state=_random_state, _remove_mean=_remove_mean, _preproc=_preproc)
	if _path is not None:
		SVM.save_to_file(_path)
	print(f"Training accuracy: {SVM.score(X_train, y_train)}")
	# print(f"Training accuracy: {score_SVM(SVM, X_train, y_train)}")
	print(f"Testing accuracy: {SVM.score(X_test, y_test)}")
	# print(f"Testing accuracy: {score_SVM(SVM, X_test, y_test)}")
	stop_watch(prog_time_start)

def display_config(kwargs):
	print_N_log(msg=f'Config:', log_dst='info')
	for key, elem in kwargs.items():
		print_N_log(msg=f"\t{key}: {elem}", log_dst='info')

def sample_Z_score(x):
	return (x-x.mean(axis=0, keepdims=True))/x.std(axis=0, keepdims=True)

def identity(x):
	return x

def pth_svm_raw_dataset_for_n_back_diff_exp_agg(dataset_name, _test_size=.33, _path=None, _remove_mean=False, _preproc=lambda x:x, _epochs=1000, _lr=1e-3, lr_decay=None, _weight_decay=0., _margin=1., _seed=0, _cuda=True):
	prog_time_start = time.time()
	logging.basicConfig(filename=_path[:-4]+'.log', level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s') #for python>3.9P: encoding='utf-8'
	display_config({'dataset_name':dataset_name, '_test_size':_test_size, '_path':_path, '_remove_mean':_remove_mean,
			'_preproc':_preproc, '_epochs':_epochs, '_lr':_lr, 'lr_decay':lr_decay, '_weight_decay':_weight_decay,
			'_margin':_margin, '_seed':_seed, '_cuda':_cuda})
	torch.save({'dataset_name':dataset_name, '_test_size':_test_size, '_path':_path, '_remove_mean':_remove_mean,
			'_preproc':_preproc, '_epochs':_epochs, '_lr':_lr, 'lr_decay':lr_decay, '_weight_decay':_weight_decay,
			'_margin':_margin, '_seed':_seed, '_cuda':_cuda}, _path[:-4]+'.config')
	init_seeds(_seed)
	_dataset = n_back_dataset(dataset_name=dataset_name, _test_size=_test_size,
		preproc_fct=lambda x: x, _remove_mean=_remove_mean, _preproc=_preproc,
		batch_size=16, test_batch_size=128, train_transfo='default', test_transfo='default',
		_random_state=_seed, _cuda=True)
	print_N_log(msg=f'{_dataset.train_loader.dataset[0][0].shape[0]}', log_dst='debug')
	# exit()
	# _SVM = pth_SVM(input_size=_dataset.train_loader.dataset[0][0].shape[0], output_size=1, reduction='none')
	_SVM = pth_SVM(input_size=_dataset.train_loader.dataset[0][0].shape[0], output_size=1, reduction='mean', margin=_margin)
	run_a_model(_model=_SVM, _dataset=_dataset, _optimer_fct=partial(torch.optim.Adam, lr=_lr, weight_decay=_weight_decay), _loss=_SVM.loss, _epochs=_epochs,  _cuda=_cuda, lr_decay=lr_decay)
	# run_a_model(_model=_SVM, _dataset=_dataset, _optimer_fct=partial(torch.optim.SGD, lr=_lr, weight_decay=_weight_decay), _loss=_SVM.loss, _epochs=_epochs,  _cuda=_cuda)
	torch.save(_SVM.state_dict(), _path)
	test_a_model(_model=_SVM, _dataset=_dataset, _loss=_SVM.loss, _cuda=_cuda)
	stop_watch(prog_time_start)
	logging.shutdown()

def pth_svm_raw_dataset_for_n_back_diff_exp_agg_Load_N_test(config_file):
	prog_time_start = time.time()
	_config = torch.load(config_file)
	display_config(_config)
	dataset_name = _config['dataset_name']
	_test_size = _config['_test_size']
	_path = _config['_path']
	_remove_mean = _config['_remove_mean']
	_preproc = _config['_preproc']
	_margin = _config['_margin']
	_seed = _config['_seed']
	_cuda = _config['_cuda']
	init_seeds(_seed)
	_dataset = n_back_dataset(dataset_name=dataset_name, _test_size=_test_size,
		preproc_fct=lambda x: x, _remove_mean=_remove_mean, _preproc=_preproc,
		batch_size=16, test_batch_size=128, train_transfo='default', test_transfo='default',
		_random_state=_seed, _cuda=True)
	print_N_log(msg=f'{_dataset.train_loader.dataset[0][0].shape[0]}', log_dst='debug')
	_SVM = pth_SVM(input_size=_dataset.train_loader.dataset[0][0].shape[0], output_size=1, reduction='mean', margin=_margin)
	_SVM.load_state_dict(torch.load(_path))
	test_a_model(_model=_SVM, _dataset=_dataset, _loss=_SVM.loss, _cuda=_cuda)
	print_N_log(f"_SVM.state_dict(): {_SVM.state_dict()}")
	stop_watch(prog_time_start)

# SVM on raw initial data for accuracy differentiation
# SVM on HRF for n-back differentiation
# SVM on HRF for accuracy differentiation
# SVM on extended time (HRF-like) but raw for n-back differentiation
# SVM on extended time (HRF-like) but raw for accuracy differentiation

#Convolution net (WaveNet)?
#Transformers?
if __name__ == '__main__':
	# svm_raw_dataset_for_n_back_diff_exp(dataset_name='HCP2021WM', _path=RESULT_DIR+'/SVM_params0')
	# thunder_svm_raw_dataset_for_n_back_diff_exp(dataset_name='HCP2021WM', _path=RESULT_DIR+'/thunder_SVM_params0')
	# thunder_svm_raw_dataset_for_n_back_diff_exp(dataset_name='HCP2021WM', _path=RESULT_DIR+'/thunder_SVM_params1', _remove_mean=True)
	# thunder_svm_raw_dataset_for_n_back_diff_exp(dataset_name='HCP2021WM', _path=RESULT_DIR+'/thunder_SVM_params2', _remove_mean=True, _preproc=lambda x: (x-x.mean(axis=0, keepdims=True))/x.std(axis=0, keepdims=True))
	# thunder_svm_raw_dataset_for_n_back_diff_exp_agg(dataset_name='HCP2021WM', _path=RESULT_DIR+'/thunder_SVM_params_agg0', _remove_mean=True, _preproc=lambda x: (x-x.mean(axis=0, keepdims=True))/x.std(axis=0, keepdims=True))
	_seed =  rand=int.from_bytes(os.urandom(4), sys.byteorder)
	# pth_svm_raw_dataset_for_n_back_diff_exp_agg(dataset_name='HCP2021WM', _test_size=.33, _path=RESULT_DIR+'/pth_SVM_agg0.pth',
	# 					_remove_mean=False, _preproc=lambda x:x, _epochs=1000,  _lr=1e-6, _weight_decay=1., _seed=_seed, _cuda=True) #1e-10 SGD
	##### Pytorch SVMs #####
	# pth_svm_raw_dataset_for_n_back_diff_exp_agg(dataset_name='HCP2021WM', _test_size=.33, _path=RESULT_DIR+'/pth_SVM_agg1.pth',
	# 					_remove_mean=False, _preproc=identity, _epochs=1000,  _lr=1e-4, lr_decay=.1, _weight_decay=1., _seed=_seed, _cuda=True) #1e-10 SGD
	# pth_svm_raw_dataset_for_n_back_diff_exp_agg(dataset_name='HCP2021WM', _test_size=.33, _path=RESULT_DIR+'/pth_SVM_agg2.pth',
	# 					_remove_mean=False, _preproc=identity, _epochs=1000,  _lr=1e-4, lr_decay=.1, _weight_decay=1., _margin=5., _seed=_seed, _cuda=True) #1e-10 SGD
	# pth_svm_raw_dataset_for_n_back_diff_exp_agg(dataset_name='HCP2021WM', _test_size=.33, _path=RESULT_DIR+'/pth_SVM_agg3.pth',
	# 					_remove_mean=True, _preproc=sample_Z_score, _epochs=1000,  _lr=1e-1, lr_decay=.1, _weight_decay=1., _seed=_seed, _cuda=True) #1e-10 SGD
	########################
	# torch.save({'dataset_name':'HCP2021WM', '_test_size': 0.33, '_path':'./results/pth_SVM_agg1.pth', '_remove_mean':False,
	# 		'_preproc':identity, '_epochs':1000, '_lr':0.0001, 'lr_decay':0.1, '_weight_decay':1.0,
	# 		'_margin':1.0, '_seed':1057243182, '_cuda':True}, RESULT_DIR+'/pth_SVM_agg1.config')

	# torch.save({'dataset_name':'HCP2021WM', '_test_size':0.33, '_path':'./results/pth_SVM_agg2.pth', '_remove_mean':False,
	# 		'_preproc':identity, '_epochs':1000, '_lr':0.0001, 'lr_decay':0.1, '_weight_decay':1.0,
	# 		'_margin':5.0, '_seed':2078604819, '_cuda':True}, RESULT_DIR+'/pth_SVM_agg2.config')

	# torch.save({'dataset_name':'HCP2021WM', '_test_size':0.33, '_path':'./results/pth_SVM_agg3.pth', '_remove_mean':True,
	# 		'_preproc':sample_Z_score, '_epochs':1000, '_lr':0.1, 'lr_decay':0.1, '_weight_decay':1.0,
	# 		'_margin':1.0, '_seed':2078604819, '_cuda':True}, RESULT_DIR+'/pth_SVM_agg3.config')

	pth_svm_raw_dataset_for_n_back_diff_exp_agg_Load_N_test(config_file=RESULT_DIR+'/pth_SVM_agg3.config')

	# pth_svm_raw_dataset_for_n_back_diff_exp_agg(dataset_name='HCP2021WM', _test_size=.33, _path=RESULT_DIR+'/pth_SVM_agg1.pth', _remove_mean=True, _preproc=lambda x: (x-x.mean(axis=0, keepdims=True))/x.std(axis=0, keepdims=True), _epochs=1000, _seed=0, _cuda=True)
	#have a look at the raw data to see if some patterns appear
	pass
#Margin? Weight decay? seed? How is the thundersvm version? What about a softmax with cross-entropy?
# z-score are contaminated cause on the statistics of both train and test data...-> TODO: modify that part to avoid contamination
