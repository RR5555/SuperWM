from time import time
import torch
import sys
import functools
import logging

def stop_watch(prog_time_start):
	"""Display the time difference between **prog_time_start** and the current time reported by :xref:time:`time.time() <>`.

	Args:
		* **prog_time_start** (:xref:float:`float <>`): A float returned by a previous call to :xref:time:`time.time() <>`
	"""
	prog_time_stop = time()-prog_time_start
	print_N_log(f'{prog_time_stop}s')
	print_N_log(f"{int(prog_time_stop/3600)}h{int(prog_time_stop/60-int(prog_time_stop/3600)*60)}m{int(prog_time_stop-int(prog_time_stop/60-int(prog_time_stop/3600)*60)*60)}s")


def progbar(curr, total, full_progbar=40, prefix_info='', suffix_info=''):
	"""Display in-place a progress bar in the ``sys.__stdout__`` of size **full_progbar**, and filled by '#' symbols up to **curr**/**total** percent of the bar.
	Can also display some info before (**prefix_info**) and after (**suffix_info**) the progress bar on the same line.

	Args:
		* **curr** (:xref:int:`int <>` or :xref:float:`float <>`): the current state of progress in the same unit and scale as **total**
		* **total** (:xref:int:`int <>` or :xref:float:`float <>`): the maximum of the progress that can be made
		* **full_progbar** (:xref:int:`int <>`, Optional): the size in number of characters that the progress bar will take. Default: 40
		* **prefix_info** (:xref:str:`str <>`): Additional pieces of information to be displayed right before the progress bar on the same line. Default: ''
		* **suffix_info** (:xref:str:`str <>`): Additional pieces of information to be displayed right after the progress bar on the same line. Default: ''
	"""
	frac = curr/total
	filled_progbar = round(frac*full_progbar)
	print('\r'+'{}['.format(prefix_info)+'#'*filled_progbar + '-'*(full_progbar-filled_progbar) + ']({:>7.2%}){}'.format(frac, suffix_info), end='', file=sys.__stdout__)

def make_one_hot(labels, C=10):
	"""Convert a tensor **labels** to a one-hot version of itself provided the number of classes **C**.

	Args:
		* **labels** (:xref:tensor:`tensor <>`): A shape (k,) or (k,1) int tensor containing ints between 0 and **C**-1 designating the labels
		* **C** (:xref:int:`int <>`): Number of classes or types of outputs

	Returns:
		* **_target** (:xref:tensor:`tensor <>`): A shape (k, **C**) 0. and 1. tensor which one hot encodes the **labels** on the same device
	"""
	_batch_size = labels.size()[0]
	#print('In make_one_hot: ', labels, flush=True)
	one_hot = torch.FloatTensor(_batch_size, C).zero_()
	# if labels.is_cuda:
	# 	one_hot = one_hot.cuda()
	one_hot = one_hot.to(labels.device)
	_target = one_hot.scatter_(1, labels.unsqueeze(1), 1)
	return _target

def flatten_data(data):
	"""Flatten a tensor while preserving the batch dimension (dim 0).

	Args:
		* **data** (:xref:tensor:`tensor <>`): A shape (k,...) tensor

	Returns:
		* (:xref:tensor:`tensor <>`) A shape (k, m) tensor flattened version of **data** where m is simply the product of all the dimensions except for the first, namely k
	"""
	_batch_size = data.size()[0]
	return data.contiguous().view(_batch_size, -1)

def init_displayed_lists(training_step_list, training_error_list, testing_error_list):
	""" Flushing the lists

	Args:
		* **training_step_list** (:xref:list:`list <>`): step list
		* **training_error_list** (:xref:list:`list <>`): training error list
		* **testing_error_list** (:xref:list:`list <>`): testing error list

	"""
	for _list in (training_step_list, training_error_list, testing_error_list):
		assert isinstance(_list, (list,tuple))
		flush_list(_list=_list)

def print_displayed_lists(training_step_list, training_error_list, testing_error_list):
	""" Displaying the lists

	Args:
		* **training_step_list** (:xref:list:`list <>`): step list
		* **training_error_list** (:xref:list:`list <>`): training error list
		* **testing_error_list** (:xref:list:`list <>`): testing error list

	"""
	print_N_log('Displayed lists:')
	print_N_log(f'training_step_list: {training_step_list}')
	print_N_log(f'training_error_list: {training_error_list}')
	print_N_log(f'testing_error_list: {testing_error_list}')

def flush_list(_list):
	"""	Empty the pointed list object, thus any shared list pointers are flushed too.

	Args:
		* _list (list): list to be flushed

	.. note::
		* Using del a[:] clears the existing list, which means anywhere it's referenced will become an empty list.
		* Using a = [] sets a to point to a new empty list, which means that other places the original list is referenced will remain non-empty.

		( https://stackoverflow.com/questions/30561194/what-is-the-difference-between-del-a-and-a-when-i-want-to-empty-a-list-c )
	"""
	del _list[:]
	#for python3.2+, can be replaced by list.clear()

def init_test_tmp_var(is_cuda=False): #TODO: check if it necessary to have that as tensors at the init
	test_loss = torch.zeros(1, requires_grad=False)
	correct = torch.zeros(1, requires_grad=False)
	if is_cuda:
		test_loss = test_loss.cuda()
		correct = correct.cuda()

	return test_loss, correct

def scale_N_display_tmp_var(legend, correct, test_loss, length_dataset):
	correct = correct.cpu().item()
	test_loss /= length_dataset
	test_loss = test_loss.cpu().item() if isinstance(test_loss, torch.Tensor) else test_loss
	percentage_correct = 100. * float(correct) / float(length_dataset)
	print_N_log('{}: Average loss: {:.4f}, Accuracy: {:.0f}/{:.0f} ({:.2f}%)\n'.format(
		legend, test_loss, correct, length_dataset, percentage_correct))
	return test_loss, percentage_correct

def update_correct(correct, output, target):
	pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
	correct += pred.eq(target.data.view_as(pred)).float().sum()

def update_loss_N_correct(test_loss, correct, output, target, loss_criterion, target_one_hot=None):
	# print_type_N_size(x_name='test_loss', x=test_loss)
	# print_type_N_size(x_name='output', x=output)
	# print_type_N_size(x_name='target', x=target)
	# print_type_N_size(x_name='target_one_hot', x=target_one_hot)
	if target_one_hot is None:
		target_one_hot = target
	# long had been added and might break stuffs
	test_loss += loss_criterion(output, target_one_hot.long()) # sum up batch loss
	pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
	correct += pred.eq(target.data.view_as(pred)).float().sum()

def with_no_grad(fct):
	_header = '''``@with_no_grad``

	'''
	fct.__doc__ = _header + fct.__doc__ if fct.__doc__ is not None else ''
	@functools.wraps(fct)
	def _with_no_grad(*args, **kwargs):
		with torch.no_grad():
			results = fct(*args, **kwargs)
		return results

	return _with_no_grad

def print_N_log(msg, log_dst='info'):
	log_dsts = {'info':logging.info, 'debug':logging.debug, 'warning':logging.warning, 'error':logging.error}
	print(msg)
	try:
		log_dsts[log_dst](msg)
	except:
		print('(No logger)')
