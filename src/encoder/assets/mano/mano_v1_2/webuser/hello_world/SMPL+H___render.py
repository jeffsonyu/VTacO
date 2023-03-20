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
- OpenCV [http://opencv.org/downloads.html] 
  --> (alternatively: matplotlib [http://matplotlib.org/downloads.html])


About the Script:
=================
This script demonstrates loading the SMPL+H model and rendering it using OpenDR 
to render and OpenCV to display (or alternatively matplotlib can also be used
for display, as shown in commented code below). 

This code shows how to:
  - Load the SMPL+H model
  - Edit pose & shape parameters of the model to create a new body in a new pose
  - Create an OpenDR scene (with a basic renderer, camera & light)
  - Render the scene using OpenCV / matplotlib


Running the Hello World code:
=============================
Inside Terminal, navigate to the mano/webuser/hello_world directory. You can run 
the hello world script now by typing the following:
>	python SMPL+H___render.py


'''

import numpy as np
from opendr.renderer import ColoredRenderer
from opendr.lighting import LambertianPointLight
from opendr.camera import ProjectPoints
from webuser.smpl_handpca_wrapper import load_model

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
m.pose[0] = np.pi

# Create OpenDR renderer
rn = ColoredRenderer()

# Assign attributes to renderer
w, h = (640, 480)

rn.camera = ProjectPoints(v=m, rt=np.zeros(3), t=np.array([0, 0.15, 1.8]), f=np.array([w,w])/2., c=np.array([w,h])/2., k=np.zeros(5))
rn.frustum = {'near': 0.5, 'far': 10., 'width': w, 'height': h}
rn.set(v=m, f=m.f, bgcolor=np.zeros(3))

# Construct point light source
rn.vc = LambertianPointLight(   f=m.f,
                                v=rn.v,
                                num_verts=len(m),
                                light_pos=np.array([-1000,-1000,-2000]),
                                vc=np.ones_like(m)*.9,
                                light_color=np.array([1., 1., 1.]))
rn.vc += LambertianPointLight(  f=m.f,
                                v=rn.v,
                                num_verts=len(m),
                                light_pos=np.array([+2000,+2000,+2000]),
                                vc=np.ones_like(m)*.9,
                                light_color=np.array([1., 1., 1.]))


# Show it using OpenCV
import cv2
cv2.imshow('render_SMPL+H', rn.r)
cv2.imwrite('./SMPL+H___hello_world___opencv.png', rn.r * 255)
print ('..Print any key while on the display window')
cv2.waitKey(0)
cv2.destroyAllWindows()

from psbody.mesh import Mesh
from psbody.mesh import MeshViewers
from psbody.mesh.sphere import Sphere
radius = .01
model_Mesh = Mesh(v=m.r, f=m.f)
model_Joints = [Sphere(np.array(jointPos), radius).to_mesh(np.eye(3)[0 if jointID == 0 else 1]) for jointID, jointPos in enumerate(m.J_transformed)]
mvs = MeshViewers(window_width=2000, window_height=800, shape=[1, 3])
mvs[0][0].set_static_meshes([model_Mesh] + model_Joints, blocking=True)
mvs[0][1].set_static_meshes([model_Mesh], blocking=True)
model_Mesh = Mesh(v=m.r, f=[])
mvs[0][2].set_static_meshes([model_Mesh] + model_Joints, blocking=True)
raw_input('Rotate the 3D viewer and press Enter to store a screenshot...')
mvs[0][0].save_snapshot('./SMPL+H___hello_world___3D_viewer.png')
raw_input('Press any Enter...')

# # Could also use matplotlib to display
# import matplotlib
# import platform
# if 'Linux' in platform.system():
#     pass  # do not need to do anything
# elif 'Darwin' in platform.system():
#     matplotlib.use("MacOSX")
# else:  
#     pass  # unhandled  # 'Windows' etc
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# from os import popen
# plt.ion()
# plt.imshow(rn.r)
# pathOUT = './SMPL+H___hello_world___matplotlib.png'
# plt.savefig(pathOUT)
# popen('open ' + pathOUT)  # OSX
# raw_input('Press any key to exit')

# # matplotlib to display vertices and joints
# import matplotlib
# import platform
# if 'Linux' in platform.system():
#     pass  # do not need to do anything
# elif 'Darwin' in platform.system():
#     matplotlib.use("MacOSX")
# else:  
#     pass  # unhandled  # 'Windows' etc
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# vertices = m.r
# joints3D = np.array(m.J_transformed).reshape((-1, 3))
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], color='r')
# ax.scatter(joints3D[:, 0], joints3D[:, 1], joints3D[:, 2], color='b')
# plt.show()
