<img width="1461" alt="ENIGMA-DTI_Normative_Modeling_logos" src="https://github.com/user-attachments/assets/c63a98e1-57be-4d16-a85c-126857cd40d2" />




This repository contains the code and examples to run normative modeling with hierarchical Bayesian regression (HBR) using the [PCNtoolkit](https://pcntoolkit.readthedocs.io/en/latest/) on white matter summary metrics derived from diffusion tensor imaging (DTI), i.e., fractional anisotropy (FA), mean diffusivity (MD), radial diffusivity (RD) and axial diffusivity (AD). A detailed description of how to process the diffusion MRI data with the ENIGMA-DTI protocol to extract white matter derivatives can be found here: [ENIGMA-DTI Protocols](https://enigma.ini.usc.edu/protocols/dti-protocols/) & [ENIGMA-DTI-TBSS Protocol](https://github.com/ENIGMA-git/ENIGMA-DTI-TBSS-Protocol). This code was used to develop and deploy the normative models presented in this paper: [Lifespan Normative Modeling of Brain Microstructure][1]

The code has been tested on PCNtoolkit v0.29 with Python 3.9, and PCNtoolkit v0.35 with Python 3.12. The instructions to install the PCNtoolkit and approximate installation times can be found here: [PCNtoolkit installation](https://github.com/amarquand/PCNtoolkit). We recommend installing PCNtoolkit and its dependencies with `pip`.

We provide an interactive cloud environment powered by Binder, so you can run all notebooks in this repository without installing anything locally. Click the button below to launch the environment:

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/villalonreina/ENIGMA-DTI-Normative-Modeling/HEAD)

Because if memory issues in Binder, we did not add PCNtoolkit directly, only its dependencies. Please make sure to install it once Binder has launched the Jupyter Notebook environment. Just type `%pip install -q pcntoolkit==0.35` before running the code to train and transfer normative models.


If you decide to install PCNtoolkit with all dependencies locally, these are the options:

## System Requirements for PCNtoolkit v0.29
The code in this repo requires the following:
- [python](https://www.python.org/) (3.9)
- [numpy] (https://pypi.org/project/numpy/1.26.4/) (1.26.4)
- [scipy] (https://pypi.org/project/scipy/1.12.0/) (1.12.0)
- [pymc] (https://pypi.org/project/pymc/5.12.0/) (5.12)
- [pytensor] (https://pypi.org/project/pytensor/2.19.0/) (2.19)

## System Requirements for PCNtoolkit v0.35
The code in this repo requires the following:
- [python](https://www.python.org/) (3.12)
- [numpy] (https://pypi.org/project/numpy/1.26.4/) (1.26.4)
- [scipy] (https://pypi.org/project/scipy/1.12.0/) (1.12.0)
- [pymc] (https://pypi.org/project/pymc/5.18.0/) (5.18)
- [pytensor] (https://pypi.org/project/pytensor/2.25.5/) (2.25.5)

## Demo of model development
Here we provide a subsample of the original dataset used in the paper of 5,697 subjects for DTI-FA. This file is in the `data` folder. One can run the training code on Ipython the following way:
```
mkdir /path/to/output_directory/norm_model/
cd code/

run nm_hbr_controls1_rob_spline_age_sexbatch_v29.py
    -controls '../data/all_sites_subsample_n5697.csv'
    -dirO '/path/to/output_directory/norm_model/'
    -age_column age
    -site_column Protocol
    -sex_column sex
    -outscaler 'standardize'
```
The options for `age_colum`, `site_column` and `sex_column` are defined by the names of the corresponding columns in the input CSV file. This code should finish running in ~1 hour with the sample data provided. There are 10 different training codes, called "nm_hbr_controls1_rob_spline_age_sexbatch_v29.py", "nm_hbr_controls2_rob_spline_age_sexbatch_v29.py", etc. Each of these is for a different train-test split (80%-20%).
When finished running, the output folder should contain a folder for each white matter region (ROI) and text CSV files containing covariates and subjects used for training and testing and evaluation metrics (i.e., MSLL, EV, SMSE, RMSE, Rho) per region and per site. 
```
ACR
ALIC
Average
BCC
CGC
CGH
CST
EC
FX
FXST
GCC
PCR
PLIC
PTR
RLIC
SCC
SCR
SFO
SLF
SS
TAP
UNC
subjects_te.txt
subjects_tr.txt
batch_te.pkl
batch_te.txt
batch_tr.pkl
batch_tr.txt
hbr_controls_metrics_f1.csv
hbr_controls_site_metrics_f1.csv
```
The folder for each white matter region should contain the Z-scores, the predictive mean (yhat) and variance (ys2), and a `Models` folder with the parameters of the normative model for that white matter region (ROI):
```
meta_data.md  NM_0_0_estimate.pkl
```

## Demo of model adaptation
We adpated the trained normative models to other datasets via *model transfer*, which is a function available in PCNtoolkit. In order to run this step you must have run the model training first and the trained normative model should be saved in the output directory specified above with the `-dirO` flag). The model adaptation script calls the estimated normative model (`meta_data.md  NM_0_0_estimate.pkl`) and saves the transfer model in another folder (also specified with a `dirO` flag). Note: we have also made the trained models from the paper available and can be found in the `models` folder of this repository. Please email Julio Villalon at julio.villalon@ini.usc.edu if you have any questions.

You can run the model transfer function in Ipython via this command:

```
cd code
mkdir /path/to/output_dir/test_transfer
out_dir_transfer='/path/to/output_dir/test_transfer'

run nm_hbr_NIMHANS_spline_age_sexbatch_transfer.py
    -controls '../data/FA_NIMHANS_ROItable_denoised_controls_final.csv'
    -dirM '/path/to/output_dir/norm_model'
    -dirO out_dir_transfer
    -age_column age
    -site_column Protocol
    -sex_column sex
    -random_state 413624  
```
One can pick any `random_state` in order to run it 10 different times, each time for a different trained model.

[1]: <https://www.biorxiv.org/content/10.1101/2024.12.15.628527v1>
