import os

from configuration import HCP_DIR, RESULT_DIR

if not os.path.isdir(HCP_DIR):
	os.mkdir(HCP_DIR)

if not os.path.isdir(RESULT_DIR):
	os.mkdir(RESULT_DIR)

fname = "hcp_task.tgz"

# print(f'wget -qO {fname} https://osf.io/s4h8j/download/')
# print(f'tar -xzf {fname} -C {HCP_DIR} --strip-components=1')

if not os.path.exists(fname):
	os.system(f'wget -qO {fname} https://osf.io/s4h8j/download/')
	os.system(f'tar -xzf {fname} -C {HCP_DIR} --strip-components=1')

# NMA provides an atlas 
fname = f"{HCP_DIR}/atlas.npz"

if not os.path.exists(fname):
	os.system(f'wget -qO {fname} https://osf.io/j5kuc/download')
