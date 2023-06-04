import numpy as np
import matplotlib.pyplot as plt
import os, sys, time
from functools import partial
print_f = partial(print, flush=True)
from moire_tracker import *
from scipy.optimize import curve_fit

# MPI
from mpi4py import MPI
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()
root = 0


# Average 
class AVG(object):
  def __init__(self,array,msites,ilslimit):
    """
    Initializes the attributes
    """
    self.array = array
    self.msites = msites
    self.ilslimit = ilslimit


  def comp_traj(self):
    # local trajectories to compute for every rank
    trajs = self.distribute_traj(self.array.shape[0],rank,size)
    print_f("%d-th rank handles %d files"\
             %(rank,(trajs[1]-trajs[0])))
    for i in range(trajs[0], trajs[1]):
      # Track moiré, order, post-process, and 
      # dump with *_self.msites names
      Track_M(i+1,self.msites,self.ilslimit).read_lammpstraj() 


  def average(self):
    """
    Restricted: Can't be called before comp_traj
    """ 
    t_av = []
    r_t_av = []
    msd_t_av = []
    # average data for all the data
    for i in range(self.array.shape[0]):
      t_av.append(np.load(str("t_"+str(i+1)+".npy")))
      r_t_av.append(np.load(str("r_t_"+str(i+1)+".npy")))
      msd_t_av.append(np.load(str("msd_t_"+str(i+1)+".npy")))
    r_t_av = np.average(np.array(r_t_av), axis=0)
    msd_t_av = np.average(np.array(msd_t_av), axis=0)
    t_av = np.average(np.array(t_av), axis=0)
    # Save data for references
    np.save("t_av.npy", t_av) 
    np.save("r_t_av.npy", r_t_av) 
    np.save("msd_t_av.npy", msd_t_av) 
    
    # Fit the data
    slope_d, inter_d = self.linear_fit(t_av, r_t_av,\
                       str("R_"+str(self.array.shape[0])))
    print_f()
    print_f("Fitting parameters for distance vs time")
    # Time is in femto-second for lammps and 
    # distance is Angstroms;
    print_f("%f : slope (mtr/sec), %f : intercept(ang.)"\
            %(slope_d*10**5, inter_d))
    print_f()
    A, B = self.quad_fit(t_av, msd_t_av,\
                       str("MSD_"+str(self.array.shape[0])))
    print_f("Fitting parameters for MSD vs time")
    print_f("%.4f : A, %.10f : B"%(A, B))
    print_f()    


  def quadratic(self, x,A,B):
    """
    Quadratic function as a representation of the delta function
    """
    return A + (B*x*x)


  def quad_fit(self, x,y,savefig, plot=True):
    """
    Fit quadratic function
    """
    (A,B), pcov = curve_fit(self.quadratic, x,y)
    fitted = A + B*(np.array(x)**2.)
    if plot == True:
      plt.plot(x, y, color="g")
      plt.plot(x, fitted, ls="--", color="b")
      plt.savefig(savefig, dpi=200)
      plt.close()
    return A, B


  def linear_fit(self, x,y,savefig, plot=True):
    """
    Linear fit for y,x
    """
    param, cov = np.polyfit(x,y,1, cov=True)
    slope = param[0]
    intercept = param[1]
    fitted = np.array(x)*slope + intercept
    if plot == True:
      plt.plot(x, y, color="g")
      plt.plot(x, fitted, ls="--", color="b")
      plt.savefig(savefig, dpi=200)
      plt.close()
    return slope, intercept


  def distribute_traj(self, num_traj, rank, size):
    """
    *Variable* block cyclic distribution to take
    care of the load-balancing and maintain easy
    data access;
    """
    # Warning
    if num_traj < size:
      if rank == root:
        print_f("Bad Parallelization! Some processes will be idle")
        print_f("Exiting...")
      comm.Abort(1)
    # Easy integer division
    traj_own = num_traj // size
    # Check if there's a remainder
    traj_rem = num_traj % size
    # Starting point for each process as long
    # as the rank < size-traj_rem
    traj_init = rank*traj_own
    # These ranks will handle 1 extra trajectories
    if rank >= (size - traj_rem):
      traj_init = (rank*traj_own) + (rank-(size-traj_rem))
      traj_own = traj_own+1
      traj_end   = traj_init+traj_own
    traj_end   = traj_init+traj_own
    return np.array([np.intc(traj_init),np.intc(traj_end)])    

