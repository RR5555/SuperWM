from shared_configuration import INIT_CONDS

HCP_BEH_DIR = "./hcp_beh"
BEH_SUM_PATH = './HCP2021_BEH.md'



# The data shared for NMA projects is a subset of the full HCP dataset
BEH_N_SUBJECTS = 100

BEH_SUBJECT_LOC = HCP_BEH_DIR+'/subjects_list.txt'

# There are 7 tasks. Each has a number of 'conditions'
# TIP: look inside the data folders for more fine-graned conditions

EXPERIMENTS = {
    'MOTOR'      : {'cond':['lf','rf','lh','rh','t','cue']},
    'WM'         : {'cond':INIT_CONDS},
    'EMOTION'    : {'cond':['fear','neut']},
    'GAMBLING'   : {'cond':['loss','win']},
    'LANGUAGE'   : {'cond':['math','story']},
    'RELATIONAL' : {'cond':['match','relation']},
    'SOCIAL'     : {'cond':['ment','rnd']}
}
