import numpy as np
import time
from average import AVG
from functools import partial
print_f = partial(print, flush=True)

# MPI
from mpi4py import MPI
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()
root = 0

# Start timer
t1 = time.time()
if rank == root:
  print_f("""
   __  ___       _        __     __                         _         
  /  |/  /___   (_)____ _/_/ ___/ /__ __ ___  ___ _ __ _   (_)____ ___
 / /|_/ // _ \ / // __// -_)/ _  // // // _ \/ _ `//  ' \ / // __/(_-<
/_/  /_/ \___//_//_/   \__/ \_,_/ \_, //_//_/\_,_//_/_/_//_/ \__//___/
                                 /___/                               
""")

  print_f("Tracking moiré sites")
  print_f()
# Options for calling the average class 
# AVG(item1, item2, ilslimit)
## item1: Array of the indices for the lammps data files;
## item2: Number of expected moiré sites i.e., number of 
##        moiré unit-cells in the data files.
## ilslimit: Guess an interlayer separation in Angstrom from which 
##           the maximum of ILS will be computed. Saves time.
# mpi enabled
avg = AVG(np.arange(1), 9, 6.75)
avg.comp_traj()

if rank == root:
  # The averarging can be run serially as
  # it's not time consuming.
  avg.average()
  print_f("Total time taken to track the moiré sites: %f s"\
           %(time.time()-t1))
