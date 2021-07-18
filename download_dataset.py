import os

from shared_configuration import HCP_DIR, RESULT_DIR
from beh_configuration import HCP_BEH_DIR

if not os.path.isdir(HCP_DIR):
	os.mkdir(HCP_DIR)

if not os.path.isdir(HCP_BEH_DIR):
	os.mkdir(HCP_BEH_DIR)

if not os.path.isdir(RESULT_DIR):
	os.mkdir(RESULT_DIR)

fname = "hcp_task.tgz"

if not os.path.exists(fname):
	os.system(f'wget -qO {fname} https://osf.io/s4h8j/download/')
	os.system(f'tar -xzf {fname} -C {HCP_DIR} --strip-components=1')


fname = "hcp_beh_task.tgz"

if not os.path.exists(fname):
	os.system(f'wget -qO {fname} https://osf.io/2y3fw/download')
	os.system(f'tar -xzf {fname} -C {HCP_BEH_DIR} --strip-components=1')


# NMA provides an atlas 
fname = f"{HCP_DIR}/atlas.npz"

if not os.path.exists(fname):
	os.system(f'wget -qO {fname} https://osf.io/j5kuc/download')
