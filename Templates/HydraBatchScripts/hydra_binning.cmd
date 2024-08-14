# @ shell=/bin/bash
#
# Sample script for LoadLeveler
#
# @ error = job.err.$(jobid)
# @ output = job.out.$(jobid)
# @ job_type = parallel
# @ node_usage= not_shared
# @ node = 1
# @ tasks_per_node = 1
# @ resources = ConsumableCpus(1)
# @ network.MPI = sn_all,not_shared,us
# @ wall_clock_limit = 23:59:59
# @ notification = complete
# @ notify_user = antti.snicker@aalto.fi
# @ queue

# run the program

#cd /ptmp/${USER}/
module purge
source /u/${USER}/WKBEAM/WKBeam-setup-hydra
poe python /u/${USER}/WKBEAM/WKBeam.py bin /u/${USER}/WKBEAM/StandardCases/ITER/ITER0fluct_XZ.txt