import numpy as np
import torch
from copy import deepcopy

from base_fct import print_N_log, progbar, init_test_tmp_var,\
	update_loss_N_correct, scale_N_display_tmp_var,\
	with_no_grad, update_correct, init_displayed_lists, print_displayed_lists,\
	print_N_log

########################## BASIC TRAINING & TESTING FCTS ###############################

#In d_t : 'model', 'train_loader', 'optimizer', 'tr_loss_criterion', 'cuda'
def train(d_t, epoch, lr_decay=None):
	print_N_log('Training epoch: {}'.format(epoch))
	# if lr_decay is not None and (epoch+1)%100==0:
	if lr_decay is not None and (epoch+1)%100==0:
		for group in d_t['optimizer'].param_groups:
			print_N_log(f"Before: {group['lr']}")
			group['lr'] = group['lr']*lr_decay
			print_N_log(f"After: {group['lr']}")
	_temp = []
	_, correct = init_test_tmp_var(is_cuda=d_t['cuda'])
	d_t['model'].train()
	for batch_idx, (data, target) in enumerate(d_t['train_loader']):
		if d_t['cuda']:
			data, target = data.cuda(), target.cuda()
		d_t['optimizer'].zero_grad()
		output = d_t['model'](data)
		loss = d_t['tr_loss_criterion'](output, target)
		###
		# sanity_loss = deepcopy(d_t['tr_loss_criterion'])
		# print(f"output: {output.squeeze()}, target: {target.squeeze()}, loss: {sanity_loss(output, target).squeeze()}")
		# print(f'data: {data}')
		# print(d_t['model'].fc.weight)
		###
		loss.backward()
		d_t['optimizer'].step()
		###
		# print(d_t['model'].fc.weight)
		# exit()
		###
		progbar(batch_idx, len(d_t['train_loader']))
		_temp.append(loss.item())
		update_correct(correct=correct, output=output, target=target)
	progbar(len(d_t['train_loader']), len(d_t['train_loader']))
	#print('\tLoss: {:.6f}'.format(loss.item()))
	_temp = np.asarray(_temp)
	_temp = np.mean(_temp)
	# print('\tLoss: {:.6f}'.format(_temp))
	_, percentage_correct = scale_N_display_tmp_var(legend='\n\tTrain set', correct=correct,
								test_loss=_temp*len(d_t['train_loader'].dataset),
								length_dataset=len(d_t['train_loader'].dataset))
	if np.isnan(_temp):
		raise ValueError('Loss is NaN')
	return _temp

#In d_t : 'model', 'test_loader', 'te_loss_criterion', 'cuda'
@with_no_grad
def test(d_t):
	print_N_log('\nTest of model :')
	d_t['model'].eval()

	test_loss, correct = init_test_tmp_var(is_cuda=d_t['cuda'])

	for batch_idx, (data, target) in enumerate(d_t['test_loader']):
		if d_t['cuda']:
			data, target = data.cuda(), target.cuda()
		output = d_t['model'](data)
		update_loss_N_correct(test_loss=test_loss, correct=correct, output=output, target=target, loss_criterion=d_t['te_loss_criterion'])
		progbar(batch_idx, len(d_t['test_loader']))
	progbar(len(d_t['test_loader']), len(d_t['test_loader']))

	test_loss, percentage_correct = scale_N_display_tmp_var(legend='\n\tTest set', correct=correct, test_loss=test_loss, length_dataset=len(d_t['test_loader'].dataset))
	return test_loss, percentage_correct

# args, train_arg_dict, test_arg_dict, training_step_list, training_error_list, testing_error_list
def train_through_epochs(epochs, train_arg_dict, test_arg_dict, training_step_list, training_error_list, testing_error_list, lr_decay=None):
	#if len(training_step_list) > 0:
	if training_step_list:
		last_epoch = training_step_list[-1]
	else:
		last_epoch = 0

	for epoch in range(1, epochs + 1):
		training_step_list.append(epoch+last_epoch)
		training_error_list.append(train(train_arg_dict, epoch+last_epoch, lr_decay))
		testing_error_list.append(test(test_arg_dict))


def run_a_model(_model, _dataset, _optimer_fct, _loss, _epochs,  _cuda=True, lr_decay=None):
	########################## 0.175: Def. accuracy error lists ###############################

	training_step_list = []
	training_error_list = []
	testing_error_list = []

	########################## 1.6: Pass model to gpu, and build optimizer ###############################
	if _cuda:
		_model.cuda()

	tr_loss_criterion = _loss
	te_loss_criterion = _loss
	print_N_log('tr_loss_criterion:{}'.format(tr_loss_criterion))

	optimizer = _optimer_fct(_model.parameters())
	print_N_log(f'optimizer: {optimizer.state_dict}')

	train_arg_dict = {'model':_model, 'train_loader':_dataset.train_loader, 'optimizer':optimizer, 'tr_loss_criterion':tr_loss_criterion, 'cuda':_cuda}
	test_arg_dict = {'model':_model, 'test_loader':_dataset.test_loader, 'te_loss_criterion':te_loss_criterion, 'cuda':_cuda}

	########################## 3: Trainings ###############################

	init_displayed_lists(training_step_list, training_error_list, testing_error_list)

	########################## Training 1 ###############################
	train_through_epochs(epochs=_epochs, train_arg_dict=train_arg_dict, test_arg_dict=test_arg_dict,
		training_step_list=training_step_list, training_error_list=training_error_list,
		testing_error_list=testing_error_list, lr_decay=lr_decay)
	print_displayed_lists(training_step_list, training_error_list, testing_error_list)

	sanity_loss = deepcopy(_loss)
	sanity_loss.reduction ='none'
	sanity_check(model=_model, train_loader=_dataset.train_loader, test_loader=_dataset.test_loader, loss_fct=sanity_loss, _cuda=True)

def test_a_model(_model, _dataset, _loss, _cuda=True):
	if _cuda:
		_model.cuda()

	tr_loss_criterion = _loss
	te_loss_criterion = _loss
	print_N_log('tr_loss_criterion:{}'.format(tr_loss_criterion))

	train_arg_dict = {'model':_model, 'test_loader':_dataset.train_loader, 'te_loss_criterion':tr_loss_criterion, 'cuda':_cuda}
	test_arg_dict = {'model':_model, 'test_loader':_dataset.test_loader, 'te_loss_criterion':te_loss_criterion, 'cuda':_cuda}

	########################## 3: Trainings ###############################
	print_N_log('Training accuracy:')
	test(d_t=train_arg_dict)
	print_N_log('Testing accuracy:')
	test(d_t=test_arg_dict)


	sanity_loss = deepcopy(_loss)
	sanity_loss.reduction ='none'
	sanity_check(model=_model, train_loader=_dataset.train_loader, test_loader=_dataset.test_loader, loss_fct=sanity_loss, _cuda=True)

def sanity_check(model, train_loader, test_loader, loss_fct, _cuda=True):
	print_N_log('Sanity check on training:')
	_data, _target = next(iter(train_loader))
	if _cuda:
		_data, _target = _data.cuda(), _target.cuda()
	output = model(_data)
	loss = loss_fct(output, _target)
	print_N_log(f"\toutput: {torch.argmax(output.squeeze(), dim=1).squeeze()}, target: {_target.squeeze()}, loss: {loss.squeeze()}")
	print_N_log('Sanity check on testing:')
	_data, _target =  next(iter(test_loader))
	if _cuda:
		_data, _target = _data.cuda(), _target.cuda()
	output = model(_data)
	loss = loss_fct(output, _target)
	print_N_log(f"\toutput: {torch.argmax(output.squeeze(), dim=1).squeeze()}, target: {_target.squeeze()}, loss: {loss.squeeze()}")
