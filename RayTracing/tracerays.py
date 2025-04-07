"""organise MPI world. Call organisation main-routine in case rank is 
a multiple of 8 and the ray-tracing-routine otherwise.
Pass the MPI world and the input data to the routine.
"""

############################################################################
# IMPORT
############################################################################

# Load standard modules
import sys
import os
import re
import glob
from mpi4py import MPI
# Load  local modules
from CommonModules.input_data import InputData 
from RayTracing.modules.mainorg import mainOrg
from RayTracing.modules.maintrace import mainTrace


############################################################################
# DRIVER FOR THE RAY TRACING PROCEDURES
############################################################################
def call_ray_tracer(input_file):

    """ Driver for the ray tracing procedures. """

    # Message passing interface for parallel computing
    comm = MPI.COMM_WORLD
    rank = comm.rank
    size = comm.size

    # For the way i/o and computing are distributed among processors
    # the size of COMM_WORLD (total number of cores) must be at least two
    if size < 2:
        raise EnvironmentError('A minimum of two CPUs are needed.')


    # Load data
    idata = InputData(input_file)

    # Available cpus are split in groups of nmbrCPUperGroup processors.
    # Within each group, one cpu (to be referred to as organization core)
    # deals with initializations and io, while the others (to be referred to 
    # as tracing cores) do the actual computations. 
    try:
        nmbrCPUperGroup = idata.nmbrCPUperGroup
    except AttributeError:
        nmbrCPUperGroup = 8
        idata.nmbrCPUperGroup = nmbrCPUperGroup

    # Perform either organization task (for organization cores) or
    # the actual ray tracing for tracing cores

    # Check first if there are already files in the output directory
    # and if so, keep them and just add the new ones afterwards
    # (this is done by adding the file index to the filename)

    # Pattern: <output_dir>/<output_filename>_file#.hdf5
    pattern = os.path.join(idata.output_dir, f"{idata.output_filename}_file*.hdf5")

    # Get list of matching files
    existing_files = glob.glob(pattern)

    # Extract numeric indices from filenames
    indices = []
    pattern_re = re.compile(f"{re.escape(idata.output_filename)}_file(\\d+)\\.hdf5")
    for file in existing_files:
        match = pattern_re.search(os.path.basename(file))
        if match:
            indices.append(int(match.group(1)))
 
    # Compute next available index
    start_index = max(indices) + 1 if indices else 0
    if rank % nmbrCPUperGroup == 0:
        mainOrg(input_file, idata, comm, start_index)
    else:
        mainTrace(idata, comm)

    # return fro the procedure
    pass
#
# END OF MAIN FUNCTION

############################################################################
# STAND-ALONE RUN
############################################################################
if __name__=='__main__':
    import sys
    input_file = sys.argv[1]
    call_ray_tracing(input_file)
#
# END OF FILE
