License:
========
To learn about MANO and SMPL+H, please visit our website: http://mano.is.tue.mpg.de
You can find the MANO/SMPL+H paper at: http://files.is.tue.mpg.de/dtzionas/MANO/paper/Embodied_Hands_SiggraphAsia2017.pdf

Visit our downloads page to download data (scans, alignments), model files and python code for MANO (hand-only) and SMPL+H (body+hands):
http://mano.is.tue.mpg.de/downloads

For comments or questions, please email us at: mano@tue.mpg.de


System Requirements:
====================
Operating system: OSX, Linux

Python Dependencies:
- Numpy & Scipy  [http://www.scipy.org/scipylib/download.html]
- Chumpy 		 [https://github.com/mattloper/chumpy]
- OpenCV 		 [http://opencv.org/downloads.html] 


Getting Started:
================

1. Extract the Code:
--------------------
Extract the "mano.zip" file to your home directory (or any other location you wish)


2. Set the PYTHONPATH:
----------------------
We need to update the PYTHONPATH environment variable so that the system knows how to find the MANO/SMPL+H code. Add the following lines to your ~/.bash_profile file (create it if it doesn't exist; Linux users might have ~/.bashrc file instead), replacing ~/mano with the location where you extracted the mano.zip (or with version 1_X: mano_v1_X.zip) file:

	MANO_LOCATION=~/mano_v1_2
	export PYTHONPATH=$PYTHONPATH:$MANO_LOCATION


Open a new terminal window to check if the python path has been updated by typing the following:
>  echo $PYTHONPATH


3. Install the 3D viewer
-------------------------------
- Please follow the installation instruction @ https://github.com/MPI-IS/mesh
- Run 'pip install opendr'   (in the same virtual environment)


4. Run the Hello World scripts:
-------------------------------
In the new Terminal window, navigate to the mano/webuser/hello_world directory. You can run the hello world scripts now by typing the following:

> python MANO___hello_world.py

OR 

> python MANO___render.py

OR 

> python SMPL+H___hello_world.py

OR 

> python SMPL+H___render.py


Note:
Both of these scripts will require the dependencies listed above. The scripts are provided as a sample to help you get started. 

Acknowledgements:
The code is based on the release code of http://smpl.is.tue.mpg.de. Therefore, we would like to kindly thank Matthew Loper and Naureen Mahmood.
