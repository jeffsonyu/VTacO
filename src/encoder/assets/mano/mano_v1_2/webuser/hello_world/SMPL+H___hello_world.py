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
This is a demo version of the script for driving the SMPL+H model with python.
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
the SMPL+H model. The code shows how to:
  - Load the SMPL+H model
  - Edit pose & shape parameters of the model to create a new body in a new pose
  - Save the resulting body as a mesh in .OBJ format


Running the Hello World code:
=============================
Inside Terminal, navigate to the mano/webuser/hello_world directory. You can run 
the hello world script now by typing the following:
>	python SMPL+H___hello_world.py

'''

from webuser.smpl_handpca_wrapper import load_model
import numpy as np

# Load SMPL+H model (here we load the female model)
m = load_model('../../models/SMPLH_female.pkl', ncomps=12, flat_hand_mean=False)

# Assign random pose and shape parameters
m.betas[:] = np.random.rand(m.betas.size) * .03
#m.pose[:] = np.random.rand(m.pose.size) * .2
m.pose[:] = [-0.17192541, +0.36310464, +0.05572387, -0.42836206, -0.00707548, +0.03556427,
             +0.18696896, -0.22704364, -0.39019834, +0.20273526, +0.07125099, +0.07105988,
             +0.71328310, -0.29426986, -0.18284189, +0.72134655, +0.07865227, +0.08342645,
             +0.00934835, +0.12881420, -0.02610217, -0.15579594, +0.25352553, -0.26097519,
             -0.04529948, -0.14718626, +0.52724564, -0.07638319, +0.03324086, +0.05886086,
             -0.05683995, -0.04069042, +0.68593617, -0.75870686, -0.08579930, -0.55086359,
             -0.02401033, -0.46217096, -0.03665799, +0.12397343, +0.10974685, -0.41607569,
             -0.26874970, +0.40249335, +0.21223768, +0.03365140, -0.05243080, +0.16074013,
             +0.13433811, +0.10414972, -0.98688595, -0.17270103, +0.29374368, +0.61868383,
             +0.00458329, -0.15357027, +0.09531648, -0.10624117, +0.94679869, -0.26851003,
             +0.58547889, -0.13735695, -0.39952280, -0.16598853, -0.14982575, -0.27937399,
             +0.12354536, -0.55101035, -0.41938681, +0.52238684, -0.23376718, -0.29814804,
             -0.42671473, -0.85829819, -0.50662164, +1.97374622, -0.84298473, -1.29958491]
# the first 66 elements correspond to body pose
# the next ncomps to left and right hand pose (ncomps/2 + ncomps/2)

# Write to an .obj file
outmesh_path = './SMPL+H___hello_world___PosedShaped.obj'
with open(outmesh_path, 'w') as fp:
    for v in m.r:
        fp.write( 'v %f %f %f\n' % ( v[0], v[1], v[2]) )

    for f in m.f+1: # Faces are 1-based, not 0-based in obj files
        fp.write( 'f %d %d %d\n' %  (f[0], f[1], f[2]) )

# Print message
print '..Output mesh saved to: ', outmesh_path 
