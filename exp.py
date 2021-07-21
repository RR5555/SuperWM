from numpy.lib.npyio import save
from svm import svm_raw_dataset_for_n_back_diff,\
	thunder_svm_raw_dataset_for_n_back_diff,\
	thunder_svm_raw_dataset_for_n_back_diff_agg
from base_fct import stop_watch
from save_N_load import save_obj
from shared_configuration import RESULT_DIR
import time
import numpy as np

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
	thunder_svm_raw_dataset_for_n_back_diff_exp_agg(dataset_name='HCP2021WM', _path=RESULT_DIR+'/thunder_SVM_params_agg0', _remove_mean=True, _preproc=lambda x: (x-x.mean(axis=0, keepdims=True))/x.std(axis=0, keepdims=True))
	#have a look at the raw data to see if some patterns appear
	pass
