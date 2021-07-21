import matplotlib
import matplotlib.pyplot as plt

from shared_configuration import RESULT_DIR
from prepare_dataset import get_raw_dataset_for_n_back_diff

def plot_dataset(X, _path):
	plt.matshow(X)
	plt.savefig(_path, format='pdf')

if __name__ == '__main__':
    _data, _labels = get_raw_dataset_for_n_back_diff(dataset_desc_dict=DATASET_DICT[dataset_name], preproc_fct=preproc_fct, include_cue=False, include_bias=True, _remove_mean=_remove_mean, _preproc=_preproc)
    plot_dataset(X, _path=RESULT_DIR+'/dataset_glimpse')