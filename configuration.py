
SUM_PATH = './HCP2021.md'


# The data shared for NMA projects is a subset of the full HCP dataset
N_SUBJECTS = 339

# There are 7 tasks. Each has a number of 'conditions'

_conds = ('0bk_body', '0bk_cor', '0bk_err', '0bk_faces', '0bk_nlr', '0bk_places', '0bk_tools',
	'2bk_body', '2bk_cor', '2bk_err', '2bk_faces', '2bk_nlr', '2bk_places', '2bk_tools',
	'all_bk_cor', 'all_bk_err', 'Sync')

_ord_conds = ('0bk_body', '0bk_faces', '0bk_places', '0bk_tools',
	'2bk_body', '2bk_faces', '2bk_places', '2bk_tools',
	'0bk_cor', '0bk_err',  '0bk_nlr', '2bk_cor', '2bk_err',  '2bk_nlr',
	'all_bk_cor', 'all_bk_err', 'Sync')

EXPERIMENTS = {
	'MOTOR'      : {'runs': [5,6],   'cond':['lf','rf','lh','rh','t','cue']},
	'WM'         : {'runs': [7,8],   'cond':_ord_conds},
	'EMOTION'    : {'runs': [9,10],  'cond':['fear','neut']},
	'GAMBLING'   : {'runs': [11,12], 'cond':['loss','win']},
	'LANGUAGE'   : {'runs': [13,14], 'cond':['math','story']},
	'RELATIONAL' : {'runs': [15,16], 'cond':['match','relation']},
	'SOCIAL'     : {'runs': [17,18], 'cond':['mental','rnd']}
}

# You may want to limit the subjects used during code development.
# This will use all subjects:
# subjects = range(N_SUBJECTS)
