# SuperWM
SuperWM (Virtual pogs/NMA-CN)

After installing the required python libs ('_pip_install.sh') in your python env or in your docker,
you can run in the project directory:
python download_dataset.py
(Downloads the dataset, the atlas, and create data dir './hcp', and result dir './results')

'exploring.py' is just the HCP2021 notebook in a file (help functions had been put in 'help_fct.py'),
and slightly modified to save results instead of showing, and applying on faces and body 0bk instead left and right hands from MOTOR.
All outputs are to be found in './results' after running 'exploring.py'.

Page 49-50: https://www.humanconnectome.org/storage/app/media/documentation/s1200/HCP_S1200_Release_Reference_Manual.pdf
differentiates between two categories BLOCKED and EVENTS, it seems that the EVENTS EV files are empty - resulting in an error for EVENT condition, tried '*nlr*'.
The condition order in the reference notebook are only including the BLOCKED conditions and that for all tasks.


