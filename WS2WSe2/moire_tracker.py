#|-----------------------------|
#|Author: Indrajit Maity       |
#|email: i.maity@imperial.ac.uk|
#|-----------------------------|

import numpy as np
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
import os, sys, time
import matplotlib.animation as animation
from functools import partial
print_f = partial(print, flush=True)

# MPI
from mpi4py import MPI
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()
root = 0


# Track Moiré Sites
class Track_M(object):
  """
  Tracks the movement of moiré sites based on 
  interlayer spacing. Each moiré sites represent AA stacking for
  twist angle close to 0 and B^X/X stacking for twist angle close
  to 60 degrees.
  """
  def __init__(self,traj_ind,msites,ilslimit):
    """
    Initialize attributes
    @input:
      traj_ind: trajectory file.
                LAMMPS format with atomic style
      msites: Number of moiré sites is extpected
              If you are doing 1x1x1 then it's 1 moiré site.
              If you are doing mxnx1 then it's m*n moiré sites.
              Used to ensure the moiré sites are extracted properly
      ilslimit: Minimum of Interlayer spacing to keep for searching
               the moiré sites.
               These ils_* are useful to reduce the search space.
    """
    self.traj_ind = traj_ind
    self.msites = msites
    self.ilslimit = ilslimit


  def mytraj(self):
    """
    @input:
      ind: index for dynamics file to be read
    @output
      Trajectory filename, string
    Multiple trajectories are used so that it takes up less memory
    and relatively easier to handle.
    """
    return str("moire_" + str(self.traj_ind))


  def get_geninfo(self, contents):
    """
    Extracts some general information
    @input
      contents: Contents/lines to be read
    @output
      ts, natom: Time series, and num of atoms
    """
    ts = []
    T = []
    ms_at_ti = []
    for i in range(len(contents)):
      if "item: timestep" in contents[i].casefold():
        ts.append(int(contents[i+1].split()[0]))
      elif "item: number of atoms" in contents[i].casefold():
        natom = int(contents[i+1].split()[0])
    return ts, natom


  def get_poscrys(self, A, pos, type1=1, type2=4):
    """
    Compute the position in crystal coordinates
    for certain type of atoms; 
    @input
      A: lattice vectors
      pos: (id, type, x, y, z) for all atoms
           5xnatom array
      type1, type2: atoms types for which ils
                    will be computed; 
                    default: 1-> Mo bottom layer
                             4-> Mo top layer
    @output
      pc_l1, pc_l2: Position in crystal coordinates
                    for the specified atom types
  """
    # Compute positions in crystal coordinate
    # This potentially is useful for gridding
    # data and computing the interlayer spacing
    n_l1 = np.sum(pos[:,1]==type1)
    n_l2 = np.sum(pos[:,1]==type2)
    pc_l1 = np.zeros((n_l1, 3), dtype=float)
    pc_l2 = np.zeros((n_l2, 3), dtype=float)

    c_l1 = int(0); c_l2 = int(0)
    for i in range(pos.shape[0]):
      # atom positions in layer 1
      if pos[i,1] == type1:
        pc_l1[c_l1] = np.dot(pos[i,2:],np.linalg.inv(A))
        c_l1 = c_l1 + 1
      # atom positions in layer 2
      elif pos[i,1] == type2:
        pc_l2[c_l2] = np.dot(pos[i,2:],np.linalg.inv(A))
        c_l2 = c_l2 + 1
    del pos
    return pc_l1, pc_l2


  def get_ils(self, pc_l1, pc_l2, A, snap):
    """
    Interlayer spacing of the moiré material for 
    a particular snapshot.
    @input
      pc_l1: positions in crystal coords for layer 1
      pc_l2: positions in crystal coords for layer 2
      A: lattice vectors
      snap: at snap-th snapshot 
    """
    # Grids for interlayer spacing interpolation
    # IM: Known ISSUES
    # ----------------
    # At low-temperature nearest neighbor works really well
    # or any of the interpolation schemes in python;
    # However, at high-temperature as the atoms fluctuate
    # quite a bit the maximum of the ILS often do not 
    # faithfully represent the moiré site. 
    # A QUICK FIX is to play with the interpolation scheme
    # and also, increase the grid-spacings (see below
    # mygrid function);
    # The dafaulted grid-spacing is 6 Angstrom. This 
    # seems to work reasonably well. 

    Na, grid_a, grid_b = self.mygrid(A)
    pc_l1_grid = griddata((pc_l1[:,0], pc_l1[:,1]), pc_l1[:,2],\
                          (grid_a, grid_b), method="nearest")
    pc_l2_grid = griddata((pc_l2[:,0], pc_l2[:,1]), pc_l2[:,2],\
                          (grid_a, grid_b), method="nearest")
    ils = (pc_l2_grid - pc_l1_grid)

    # Removes the nan
    D = []
  #  out_f = str("ils_") + str(snap) 
  #  g = open(out_f, "w")
    for i in range(Na):
      for j in range(Na):
        if np.isnan(ils[i][j]) == False:
          tmp = np.array([i/float(Na-1), j/float(Na-1),\
                   ils[i][j]])
          D.append(tmp)
#          g.write("%.6f %.6f %.6f\n"%(tmp[0], tmp[1], tmp[2]))
#    g.close()
    D = np.array(D)
    ind = np.where(D[:,2] >= self.ilslimit/A[2,2])
    a = self.msite_at_snap(A,D[ind[0]], plot=False)
    return a 


  def mygrid(self,A):
    """
    Returns a square grid for a lattice
    @input
      A: Lattice vectors
    @output
      Na: Grid-points along x (same for y)
      grid_a, grid_b: Grid
    """
    # Default is approximately 2 angstrom apart
    Na = int(np.ceil(np.linalg.norm(A[0])/6.0))
    a = np.linspace(0, 1, Na)
    b = np.linspace(0, 1, Na)
    # Gridding in crystal-coordinates
    grid_a, grid_b = np.meshgrid(a, b, indexing="ij")
    return Na, grid_a, grid_b


  def msite_at_snap(self,A,points,plot=True,check=True):
    """
    Finds all the moiré sites from a snapshot by locating 
    the maximum of the interlayer spacing.
    @input
      A: Lattice vectors
      points: Points containing x, y, ils
              (in crystal coordinates)
    @output
      Approximate locations of moiré lattice sites.
    """
    ms = []
    for i in range(points.shape[0]):
      tmp = []
      #--------------------
      # Physical intuition-
      # Two moiré sites need to be separated by the 
      # moiré lattice constant. Therefore, consider
      # all the points that are larger than the specified
      # ils and figure out which one is maximum
      # note: For multiple "maximum", keep only one of them. 
      # CAUTION:
      # The program dies if the extracted number of moiré 
      # sites do not match the number of moiré unit-cells.
      # THIS IS THE KEY SAFETY CHECK;
      # NOTE: Some on-the-fly remedies are also provided
      #       If nothing works, you need to play with grid
      #       spacing or the interlayer separation interpol
      #       ation method; See above known ISSUES;
      #---------------------------
      for j in range(points.shape[0]):
        if self.dist_c(A, points[i], points[j])\
           <= np.linalg.norm(A[0])/(np.sqrt(self.msites)*2.):
          tmp.append(points[j])
      tmp = np.array(tmp)
      ind = np.where(tmp[:,2] >= np.max(tmp[:,2]))
      ms.append(tmp[ind[0][0],:].reshape(3))

    if check == True:
      if self.msites != np.unique(ms, axis=0).shape[0]:
        print_f("Expected number of moiré sites don't match")
        print_f("Found %d sites"%(np.unique(ms, axis=0).shape[0]))
        print_f("Attempting to fix PBC on the fly")
        un = np.unique(ms, axis=0)
        index = []
        # Same physical intuition but this time try 
        # 5 angstoms as skin distance; 
        # Ad-hoc; REMOVE this soon
        for i in range(un.shape[0]):
          for j in range(i+1,un.shape[0]):
            if self.dist_c(A, un[i,:], un[j,:])\
               <= (np.linalg.norm(A[0])/(np.sqrt(self.msites)*2))+5.0:
              index.append(j)
        # Forcefully delete
        un = np.delete(un, index, 0)
        if  self.msites == un.shape[0]:
          print_f("-> Attempted Ad-hoc PBC fix succeded!!")
          return un
        else:
          print_f("Attempted PBC fix failed!!")
          print_f("Found %d sites"%(un.shape[0]))
          print_f("Exiting...")
          comm.Abort(1)
      else:
        return np.unique(ms, axis=0)


  def dist_c(self,A,p1,p2):
    """
    Computes and returns the distance between two points
    including the periodic boundary conditions in 2D.
    @input
      A: lattice vectors
      p1,p2: Points in crystal coordinates
    """
    d = p2-p1
    if np.abs(d[0]) >= 0.5:
      if p2[0] >= p1[0]:
        x2_un = p2[0] - 1.0
      else :
        x2_un = p2[0] + 1.0
    else:
      x2_un = p2[0]

    if np.abs(d[1]) >= 0.5:
      if p2[1] >= p1[1]:
        y2_un = p2[1] - 1.0
      else :
        y2_un = p2[1] + 1.0
    else:
      y2_un = p2[1]
    d = np.array([x2_un, y2_un])-p1[:2]
    dist = np.dot(d, A[:2,:2])
    return np.linalg.norm(dist)



  def unwrap(self,A,p1,p2):
    """
    Computes and returns the distance between two points
    including the periodic boundary conditions.
    @input
      p1,p2: Points in crystal coordinates
    """
    d = p2-p1
    if np.abs(d[0]) >= 0.5:
      if p2[0] >= p1[0]:
        x2_un = p2[0] - 1.0
      else :
        x2_un = p2[0] + 1.0
    else:
      x2_un = p2[0]

    if np.abs(d[1]) >= 0.5:
      if p2[1] >= p1[1]:
        y2_un = p2[1] - 1.0
      else :
        y2_un = p2[1] + 1.0
    else:
      y2_un = p2[1]

    return np.dot(p1, A), np.dot(np.array([x2_un,y2_un,p2[2]]),A)


  def read_lammpstraj(self):
    """
    read lammps trajectory
    """ 
    # Read text files and extract informations
    filename = self.mytraj()
    if rank == root:
      print_f("=====================")
      print_f("  %s  "%(filename))
      print_f("=====================")
    try:
      f = open(filename, "r")
      lines = f.readlines()
      f.close()
    except OSError:
      print_f("File format not supported")
      print_f("Exiting...")
      comm.Abort(1)
    self.ts, self.natom = self.get_geninfo(lines)
    if rank == root:
      print_f("Number of snapshots:%d in file:%s"\
              %(len(self.ts), filename))
      print_f("Number of atoms:%d in file:%s"\
              %(self.natom, filename))

    # To skip "n" snapshots
    skip = 1
    T = []
    ms_at_ti = []
    loc_moire = []

    snap_s = time.time()
    for i in range(0, len(self.ts), skip):
      # ***Hard-coded***
      # 9 is default in lammps traj file
      init = (i*(self.natom + 9))
      end = (i+1)*(self.natom + 9)

      counter = 0
      # for every snapshot compute stuff
      for j in range(init, end):
        if "box bounds" in lines[j].casefold():
          xlo = eval(lines[j+1].split()[0])
          xhi = eval(lines[j+1].split()[1])
          xy = eval(lines[j+1].split()[2])
          ylo = eval(lines[j+2].split()[0])
          yhi = eval(lines[j+2].split()[1])
          yz = eval(lines[j+2].split()[2])
          zlo = eval(lines[j+3].split()[0])
          zhi = eval(lines[j+3].split()[1])
          xz = eval(lines[j+3].split()[2])
          A = np.array([[(xhi-xlo)-xy, 0.0, 0.0],\
                        [xy, yhi-ylo, 0.0],\
                        [xz, yz, zhi-zlo]])
        # information on atoms
        elif "item: atoms" in lines[j].casefold():
          # data-for each snapshot
          p = np.empty((self.natom, 5), dtype=object)
          for k in range(j+1, j+1+self.natom):
            for l in range(2):
              p[k-j-1][l] = int(lines[k].split()[l])
            for l in range(2, 5):
              p[k-j-1][l] = eval(lines[k].split()[l])
       
          #-----------------------------
          # position in crystal coordinates
          # Only interlayer sepration between type "m",
          # and atom type "n" will be computed.
          # Below m=1,n=4
          #-----------------------------
          pc_l1, pc_l2 = self.get_poscrys(A, p, 1, 4)
          ms_at_ti.append(self.get_ils(pc_l1, pc_l2, A, self.ts[i]))
          T.append(self.ts[i])
          if rank == root:
            print_f("Snapshot %d done out of %d"%(i, len(self.ts)))
            snap_f = time.time()
            print_f("%f secs. to complete %d snapshots"\
                   %(snap_f-snap_s, i+1))
    del lines
    
    # Order the moiré lattices as they could be
    # difficult to figure out where they have moved
    ms_at_ti = np.array(ms_at_ti)
    ordered_ms = np.copy(ms_at_ti)
    # for all the snapshots
    for i in range(ms_at_ti.shape[0]-1):
      # Order the moiré sites and find out
      # the neighbor of j; 
      # Physical intuition: The closest moiré
      # site at t_i+1 (with t_i) would be the
      # same moiré site that has travelled in time
      ind = []
      for j in range(ms_at_ti.shape[1]):
        tmp = []
        for k in range(ms_at_ti.shape[1]):
          d = self.dist_c(A, ordered_ms[i,j], ms_at_ti[i+1,k])
          tmp.append(d)
        ind.append(np.argmin(tmp))
      for l in range(len(ind)):
        ordered_ms[i+1,l,:] = ms_at_ti[i+1,ind[l],:]
  
    # save a file for creating a movie
    dynamics = "DYNAMICS_" + str(self.traj_ind)
    np.save(dynamics, ordered_ms)
    
    # Time series with origin at the initial point
    t = [T[i+1]-T[0] for i in range(len(T)-1)]
    r_t = []
    d2_t = []
    rvec = np.zeros((3), dtype=float)
    for i in range(ordered_ms.shape[0]-1):
      d2_tmp = []
      r_tmp = []
      vec = []
      for j in range(ordered_ms.shape[1]):
        p1, p2_un = self.unwrap(A, ordered_ms[i,j,:], ordered_ms[i+1,j,:])
        d = np.sqrt(np.square(p2_un[0]-p1[0])+\
                              np.square(p2_un[1]-p1[1]))
        # Safety check:
        # If the distance moves more than 30 angstrom then it's
        # moving too fast or a possible chance to *bad* tracking.
        if d > 30:
          print_f("Moiré sites moves too much in every steps")
          print_f("Signs of bad moiré tracker or trajectories or")
          print_f("Maybe reduce the steps in lammps trajectroies")
          comm.Abort(1)
        vec.append([p2_un[0]-p1[0], p2_un[1]-p1[1], 0.0])
      vec = np.array(vec)
      #print_f(vec)
      #print_f("average pos")
      #print_f(np.average(vec, axis=0))
      rvec = rvec + np.average(vec, axis=0)
      #print_f("rvec", rvec)
      r_t.append(np.linalg.norm(rvec))
      d2_t.append(np.linalg.norm(rvec)**2.)

    # Save as standard binary files
    np.save(str("t_"+str(self.traj_ind)), np.array(t))
    np.save(str("r_t_"+str(self.traj_ind)), np.array(r_t))
    np.save(str("msd_t_"+str(self.traj_ind)), np.array(d2_t))
