from time import time

def stop_watch(prog_time_start):
	"""Display the time difference between **prog_time_start** and the current time reported by :xref:time:`time.time() <>`.

	Args:
		* **prog_time_start** (:xref:float:`float <>`): A float returned by a previous call to :xref:time:`time.time() <>`
	"""
	prog_time_stop = time()-prog_time_start
	print(prog_time_stop, " s", flush=True)
	print(int(prog_time_stop/3600), "h", int(prog_time_stop/60-int(prog_time_stop/3600)*60), "m", int(prog_time_stop-int(prog_time_stop/60-int(prog_time_stop/3600)*60)*60), "s", flush=True)
