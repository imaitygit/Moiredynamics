```
   __  ___       _        __     __                         _         
  /  |/  /___   (_)____ _/_/ ___/ /__ __ ___  ___ _ __ _   (_)____ ___
 / /|_/ // _ \ / // __// -_)/ _  // // // _ \/ _ `//  ' \ / // __/(_-<
/_/  /_/ \___//_//_/   \__/ \_,_/ \_, //_//_/\_,_//_/_/_//_/ \__//___/
                                 /___/                                
```

## Description
This allows you to track the movements of moiré sites obtained by
classical molecular dynamics simulations performed with the LAMMPS
package. A movie can also be created using the code.

The code computes the interlayer separation between two layers of a moiré
material for every time step and then follows the motion of the maximum of the
interlayer separation. The maximum of the interlayer separation implies one
moiré site per moiré unit cell and is associated with the most unfavourable
stacking. By tracking the movements of the moiré sites, we compute the
velocity and mean-square-displacements with time.

The examples are provided for a system containing WS2 and WSe2 (see
[WS2WSe2](./WS2WSe2) folder)

## Visuals
Coming soon...

## Installation
This code uses python3 (tested on python3.9), 
+ matplotlib, 
+ numpy,
+ scipy
+ mpi4py 

## Support
Please email me [@Imperial](mailto:i.maity@imperial.ac.uk) or
[@gmail](mailto:indrajit.maity02@gmail.com) if you have any questions or have
found any bugs.

## Authors and acknowledgment
If you use the codes for your work, consider citing the paper for which the
code was developed:
```
@misc{rossi2023phasonmediated,
      title={Phason-mediated interlayer exciton diffusion in WS2/WSe2 moir\'e heterostructure}, 
      author={Antonio Rossi and Jonas Zipfel and Indrajit Maity and Monica Lorenzon and Luca Francaviglia and Emma C. Regan and Zuocheng Zhang and Jacob H. Nie and Edward Barnard and Kenji Watanabe and Takashi Taniguchi and Eli Rotenberg and Feng Wang and Johannes Lischner and Archana Raja and Alexander Weber-Bargioni},
      year={2023},
      eprint={2301.07750},
      archivePrefix={arXiv},
      primaryClass={cond-mat.mtrl-sci}
}

``` 
