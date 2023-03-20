'''
Copyright 2017 Javier Romero, Dimitrios Tzionas, Michael J Black and the Max Planck Gesellschaft.  All rights reserved.
This software is provided for research purposes only.
By using this software you agree to the terms of the MANO/SMPL+H Model license here http://mano.is.tue.mpg.de/license

More information about MANO/SMPL+H is available at http://mano.is.tue.mpg.de.
For comments or questions, please email us at: mano@tue.mpg.de

Acknowledgements:
The code file is based on the release code of http://smpl.is.tue.mpg.de with adaptations. 
Therefore, we would like to kindly thank Matthew Loper and Naureen Mahmood.


Please Note:
============
This is a demo version of the script for driving the MANO model with python.
We would be happy to receive comments, help and suggestions on improving this code 
and in making it available on more platforms. 


System Requirements:
====================
Operating system: OSX, Linux

Python Dependencies:
- Numpy & Scipy  [http://www.scipy.org/scipylib/download.html]
- Chumpy [https://github.com/mattloper/chumpy]


About the Script:
=================
This script demonstrates a few basic functions to help users get started with using 
the MANO model. The code shows how to:
  - Load the MANO model
  - Edit pose & shape parameters of the model to create a new body in a new pose
  - Save the resulting body as a mesh in .OBJ format


Running the Hello World code:
=============================
Inside Terminal, navigate to the mano/webuser/hello_world directory. You can run 
the hello world script now by typing the following:
>	python MANO___hello_world.py

'''

from webuser.smpl_handpca_wrapper_HAND_only import load_model
import numpy as np

## Load MANO/SMPL+H model (here we load the righ hand model)
## Make sure path is correct
m = load_model('../../models/MANO_RIGHT.pkl', ncomps=6, flat_hand_mean=False)

## Assign random pose and shape parameters
m.betas[:] = np.random.rand(m.betas.size) * .03
#m.pose[:] = np.random.rand(m.pose.size) * 1.0
m.pose[:3] = [0., 0., 0.]
m.pose[3:] = [-0.42671473, -0.85829819, -0.50662164, +1.97374622, -0.84298473, -1.29958491]
# the first 3 elements correspond to global rotation
# the next ncomps to the hand pose

## Write to an .obj file
outmesh_path = './MANO___hello_world___PosedShaped.obj'
with open(outmesh_path, 'w') as fp:
    for v in m.r:
        fp.write( 'v %f %f %f\n' % ( v[0], v[1], v[2]) )

    for f in m.f+1: # Faces are 1-based, not 0-based in obj files
        fp.write( 'f %d %d %d\n' %  (f[0], f[1], f[2]) )

## Print message
print '..Output mesh saved to: ', outmesh_path 
