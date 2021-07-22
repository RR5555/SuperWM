from prepare_dataset import DATASET_DICT,\
	get_raw_dataset_for_n_back_diff_agg
from base_fct import print_N_log


import torch
import torchvision
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split

############################### DATASET CLASSES AND FCTS ###################################

# dataset_name, _test_size=.33, preproc_fct=lambda x: x, _remove_mean=False, _preproc=lambda x:x

class n_back_dataset:
	def __init__(self, dataset_name='HCP2021WM', _test_size=.33, preproc_fct=lambda x: x, _remove_mean=False, _preproc=lambda x:x, batch_size=64, test_batch_size=64, train_transfo='default', test_transfo='default', _random_state=0, _cuda=True):
		_data, _labels = get_raw_dataset_for_n_back_diff_agg(dataset_desc_dict=DATASET_DICT[dataset_name], preproc_fct=preproc_fct, include_cue=False, include_bias=False, _remove_mean=_remove_mean, _preproc=_preproc)
		X_train, X_test, y_train, y_test = train_test_split(_data, _labels, test_size=_test_size, random_state=_random_state)

		self.alias_transforms = {'default':self.default_transform, 'norm':self.norm_transform}
		if train_transfo not in self.alias_transforms:
			raise ValueError('train_transfo argument not valid for mnist dataset: got {} instead of {}'.format(train_transfo, self.alias_transforms.keys()))
		if test_transfo not in self.alias_transforms:
			raise ValueError('test_transfo argument not valid for mnist dataset: got {} instead of {}'.format(test_transfo, self.alias_transforms.keys()))
		_norm_tuple = ('norm')
		if (train_transfo in _norm_tuple and test_transfo not in _norm_tuple) or (train_transfo not in _norm_tuple and test_transfo in _norm_tuple):
			raise ValueError('If one transformation uses norm ({}) the other should too: train_transfo:{}; test_transfo:{}'.format(_norm_tuple, train_transfo, test_transfo))

		self.train_transfo = train_transfo
		self.test_transfo = test_transfo
		print_N_log(f"{self.alias_transforms[train_transfo]()(X_train).shape}, {y_train.shape}", log_dst='debug')
		print_N_log(f"{self.alias_transforms[test_transfo]()(X_test).shape}, {y_test.shape}", log_dst='debug')

		train_dataset = torch.utils.data.TensorDataset(self.alias_transforms[train_transfo]()(X_train).float(), torch.from_numpy(y_train))
		test_dataset = torch.utils.data.TensorDataset(self.alias_transforms[test_transfo]()(X_test).float(), torch.from_numpy(y_test))

		kwargs = {'num_workers': 1, 'pin_memory': True} if _cuda else {}
		self.train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, **kwargs)
		self.test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False, **kwargs)
		self.labels = ['0-back', '2-back'] # 0: 0-back, 2: 2-back

	def set_train_loader_to_test(self):
		self.train_loader.dataset.transform = self.alias_transforms[self.test_transfo]()

	def reset_loader_transfos(self):
		self.train_loader.dataset.transform = self.alias_transforms[self.train_transfo]()
		self.test_loader.dataset.transform = self.alias_transforms[self.test_transfo]()

	def get_label_from_index(self, index):
		if isinstance(index, int):
			return str(index)
		else:
			return str(index.item())

	def get_nb_classes(self):
		return len(self.labels)

	def get_labels(self):
		return self.labels

	## Transforms ##

	def default_transform(self):
		return transforms.Compose([
					transforms.ToTensor(),
					torch.squeeze
				])

	def norm(self):
		return transforms.Normalize((0.1307,), (0.3081,))

	def denorm(self):
		return transforms.Compose([transforms.Normalize(mean=[0.,], std=[1/0.3081,]),
								transforms.Normalize(mean=[-0.1307,], std=[1.,]),
								])

	def norm_transform(self):
		return transforms.Compose([
					transforms.ToTensor(),
					self.norm() #If Normalize is applied, the x_i s won't be in [0,1] anymore
				])



