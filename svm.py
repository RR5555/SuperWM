from prepare_dataset import get_raw_dataset_for_n_back_diff,\
    DATASET_DICT, get_raw_dataset_for_n_back_diff_agg

import sklearn
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from thundersvm import SVC as th_SVC

# 'HCP2021WM', 'HCP2021WM_BEH'

# SVM on raw initial data for n-back differentiation
def svm_raw_dataset_for_n_back_diff(dataset_name, _test_size=.33, _random_state=0):
	_data, _labels = get_raw_dataset_for_n_back_diff(dataset_desc_dict=DATASET_DICT[dataset_name], preproc_fct=lambda x: x, include_cue=False)
	X_train, X_test, y_train, y_test = train_test_split(_data, _labels, test_size=_test_size, random_state=_random_state)
	print(f'Training set:{X_train.shape}; testing set:{X_test.shape}')
	SVM = SVC(kernel='linear', dual=False, random_state=_random_state)
	SVM.fit(X_train, y_train)
	return SVM, X_train, X_test, y_train, y_test

# https://github.com/Xtra-Computing/thundersvm/tree/master/
def thunder_svm_raw_dataset_for_n_back_diff_agg(dataset_name, _test_size=.33, _random_state=0, preproc_fct=lambda x: x, _remove_mean=False, _preproc=lambda x:x):
	_data, _labels = get_raw_dataset_for_n_back_diff_agg(dataset_desc_dict=DATASET_DICT[dataset_name], preproc_fct=preproc_fct, include_cue=False, include_bias=True, _remove_mean=_remove_mean, _preproc=_preproc)
	X_train, X_test, y_train, y_test = train_test_split(_data, _labels, test_size=_test_size, random_state=_random_state)
	print(f'Training set:{X_train.shape}; testing set:{X_test.shape}')
	SVM = th_SVC(kernel='linear', random_state=_random_state, max_iter=int(1e3))
	SVM.fit(X_train, y_train)
	return SVM, X_train, X_test, y_train, y_test



def thunder_svm_raw_dataset_for_n_back_diff(dataset_name, _test_size=.33, _random_state=0, preproc_fct=lambda x: x, _remove_mean=False, _preproc=lambda x:x):
	_data, _labels = get_raw_dataset_for_n_back_diff(dataset_desc_dict=DATASET_DICT[dataset_name], preproc_fct=preproc_fct, include_cue=False, include_bias=True, _remove_mean=_remove_mean, _preproc=_preproc)
	X_train, X_test, y_train, y_test = train_test_split(_data, _labels, test_size=_test_size, random_state=_random_state)
	print(f'Training set:{X_train.shape}; testing set:{X_test.shape}')
	SVM = th_SVC(kernel='linear', random_state=_random_state, max_iter=int(1e3))
	SVM.fit(X_train, y_train)
	return SVM, X_train, X_test, y_train, y_test



# SVM on raw initial data for accuracy differentiation
# SVM on HRF for n-back differentiation
# SVM on HRF for accuracy differentiation
# SVM on extended time (HRF-like) but raw for n-back differentiation
# SVM on extended time (HRF-like) but raw for accuracy differentiation