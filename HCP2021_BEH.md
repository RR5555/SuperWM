Atlas path (the datasets share a same atlas downloaded in ./hcp/atlas.npz):
- hcp
	- atlas.npz (Atlas file)
Folder organisation:
- hcp_beh
	- regions.npy (information on the brain parcellation)
	- subjects_list.txt (list of subject IDs)
	- subjects (main data folder)
		- [subjectID] (subject-specific subfolder)
			- EXPERIMENT (one folder per experiment)
				- RUN (one folder per run)
					- data.npy (the parcellated time series data)
					- EVs (EVs folder)
						- [ev1.txt] (one file per condition)
						- [ev2.txt]
						- Stats.txt (behavioural data [where available] - averaged per run)
						- Sync.txt (ignore this file)
Number of subjects: 100
Exp setting:
	TR: 0.72
	_cue_duration: 2.5
	_trial_duration: 2.5
	_ITF_duration: 15
	nb_cue_per_block: 1
	nb_trial_per_block: 10
	nb_ITF_per_block: 4
	nb_block: 8
	Supposed run duration in s (run_duration): (nb_cue_per_block*_cue_duration+nb_trial_per_block*_trial_duration)*nb_block+nb_ITF_per_block*_ITF_duration=(1*2.5+10*2.5)*8+4*15=280.0
	Supposed run duration in frames: run_duration/TR=280.0/0.72=388.8888888888889
Format of the timeseries data (for subject 0, subject 0 might be different in the two datasets): (360, 405)
Run duration in s (for subject 0, subject 0 might be different in the two datasets)): 291.59999999999997

####################
Without mean substraction:
Stats stored at ./results/beh_summary_all_parcels.fth:
     min           max         range          mean
0    0.0  12711.705058  12711.705058  10788.724639
1    0.0  12568.730199  12568.730199   9785.014077
2    0.0  13351.351420  13351.351420  10834.008511
3    0.0  12790.799806  12790.799806  10837.722498
4    0.0  12280.979654  12280.979654  10467.231383
..   ...           ...           ...           ...
355  0.0  13237.364780  13237.364780  10972.947416
356  0.0  11300.725107  11300.725107   7425.891291
357  0.0  14808.020696  14808.020696  12265.323672
358  0.0  17887.033981  17887.033981  14689.323637
359  0.0  17856.938864  17856.938864  14636.945474

[360 rows x 4 columns]
Marginal stat over all parcels:
	min:0.0
	max:19815.796318812085
	max range:19815.796318812085

####################
With mean substraction:
Stats stored at ./results/beh_summary_all_parcels_w_mean_sub.fth:
             min          max        range          mean
0   -1290.098341   610.493186  1900.591527 -1.578928e-13
1   -1692.245767   764.402934  2456.648701  3.229268e-14
2   -1806.666028  1066.500278  2873.166305  1.015939e-13
3   -1189.295886   562.751674  1752.047559 -6.105966e-14
4   -1730.758205   690.768545  2421.526750 -5.467074e-14
..           ...          ...          ...           ...
355 -3070.674916   456.685852  3527.360768 -2.272614e-14
356 -2347.318175   391.433112  2738.751288 -3.274742e-14
357 -3317.521919   526.742936  3844.264855  9.775384e-14
358 -2590.946294   905.510603  3496.456897  2.335493e-14
359 -3289.972202  1144.192871  4434.165073 -2.036145e-13

[360 rows x 4 columns]
Marginal stat over all parcels:
	min:-5049.757444788872
	max:2821.231705802682
	max range:5806.397649191439
