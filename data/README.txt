This is Supplemental Information (SI) for "Detecting and tracking drift in quantum information processors" by
Proctor et al. A preprint of this paper can be found at https://arxiv.org/abs/1907.13608.

This folder contains all the data and analysis code presented in that paper. The core data analysis routines use
PyGSTi, which can be found at https://github.com/pyGSTio/pyGSTi. 

The analysis was run using pyGSTi commit 7c6ddd1de209b795ea39bfb69d010b687e812d07. This code does *not* work
on the latest full release of pyGSTi (0.9.9). It is anticipated that it will work with the next full release
of pyGSTi (0.9.10).

Below is a basic guide to navigating the folders of this SI:

-------------------------
--- ramsey/experiment ---
-------------------------
This folder contains the data and analysis code for the time-resolved Ramsey experiment, the results of which
are presented in Figure 1 of the paper. The folder contains a single Jupyter notebook, which runs all of the
data analysis.

-------------------------
----- rb/simulation -----
-------------------------
This contains the data and analysis code for the simulation of time-resolved RB, the results of which
are presented in Figure 2 of the paper. The folder contains a single Jupyter notebook, which runs all of the
data analysis on the simulated data, and which can be used to run new simulations with the same noise model.

-------------------------
---- gst/simulation -----
-------------------------
This contains the data and analysis code for the simulation of time-resolved GST, the results of which 
are presented in Figure 2 of the paper. The raw simulated data is contained in the "data" folder. All the
code is contained in the "analysis" folder. This contains the following code files:

- create_simulated_data.py : this generates the simulated data. **This was run using MPI on 20 cores**.
- drift.ipynb : this contains the general circuit-agnostic drift analysis.
- trgst_fit.py : this contains the TR-GST model-fitting code. **This was run using MPI on 20 cores**.
- tdmodel.py : encodes the general time-dependent model that the data is fit to.

--------------------------
---- gst/experiments -----
--------------------------
This contains the data and analysis code for the two time-resolved GST experiments, the results of which
are presented in Figure 3 of the paper. The raw data is contained in the two folders "data/1" and "data/2", 
corresponding to the first and second experiment, respectively. All analysis code is contained in the
"analysis" folder. This contains the following code files:

- drift.ipynb : this contains the general circuit-agnostic drift analysis.
- gst.ipynb : this contains the standard GST analysis, used to inform the TR-GST analysis.
- trgst_fit.py : this contains the TR-GST model-fitting code. **This was run using MPI on 20 cores**.
- trgst_plotting.ipnyb : this contains code that analyzes the results of the TR-GST fit.
- tdmodel.py : encodes the general time-dependent model that the data is fit to.

