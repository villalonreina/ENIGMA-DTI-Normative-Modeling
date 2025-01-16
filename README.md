<img width="1461" alt="ENIGMA-DTI_Normative_Modeling_logos" src="https://github.com/user-attachments/assets/c63a98e1-57be-4d16-a85c-126857cd40d2" />




This repository contains the code to run normative modeling with hierarchical Bayesian regression (HBR) using the [PCNtoolkit](https://pcntoolkit.readthedocs.io/en/latest/) on white matter summary metrics derived from diffusion tensor imaging (DTI), i.e., fractional anisotropy (FA), mean diffusivity (MD), radial diffusivity (RD) and axial diffusivity (AD). A detailed description of how to process the diffusion MRI data with the ENIGMA-DTI protocol to extract white matter derivatives can be found here: [ENIGMA-DTI Protocols](https://enigma.ini.usc.edu/protocols/dti-protocols/). This code was used to develop and deploy the normative models presented in this paper: [Lifespan Normative Modeling of Brain Microstructure][1]

## System Requirements
The code in this repo requires the following:
- [python](https://www.python.org/) (>= 3.9)
- [PCNtoolkit](https://pcntoolkit.readthedocs.io/en/latest/) (>= 0.29)

The code has been tested on Python 3.9 and PCNtoolkit v29. The instructions to install the PCNtoolkit and approximate installation times can be found here: [PCNtoolkit installation](https://github.com/amarquand/PCNtoolkit).

## Demo of model development
Here we provide a subsample of the original dataset used in the paper of ~5400 subjects for DTI-FA. This file is in the `data` folder. One can run the training code on Ipython the folowing way:
```
mkdir /path/to/output_dir/norm_model
out_dir=/path/to/output_dir/norm_model
cd code/

run nm_hbr_controls1_rob_spline_age_sexbatch_v29.py
    -controls '../data/all_sites_subsample_n5459.csv'
    -dirO {out_dir}
    -age_column age
    -site_column Protocol
    -sex_column sex
    -outscaler 'standardize'
```
The options for `age_colum`, `site_column` and `sex_column` are defined by the names of the corresponding columns in the input CSV file. This code should finish running in ~1 hour with the sample data provided.
When finished running, the output folder should contain a folder for each white matter region (ROI) and text CSV files containing covariates and subjects used for training and testing and evaluation metrics (i.e., MSLL,EV,SMSE,RMSE,Rho) per region and per site. 
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
The folder for each white matter region should contain the Z-scores, the predictive mean (yhat) and variance (ys2), and a `Models` folder with the parameters of the normative model for that region:
```
meta_data.md  NM_0_0_estimate.pkl
```

## Demo of model adaptation
We adpated the trained normative models to other datasets via *model transfer*, which is capability available in the PCNtoolkit. In order to run this step you must have run the model training first and the model should be saved in the outout directory previously specified (see above). This script calls the estimated model and saves the transfer model in another folder. Note: The data used for this step in the paper cannot be made public and is available upon reasonable request. Please email Julio Villalon at julio.villalon@ini.usc.edu.

You can run the model transfer function in Ipython via this command:

```
cd code
mkdir /path/to/output_dir/test_transfer
out_dir_transfer=/path/to/output_dir/test_transfer

run nm_hbr_NIMHANS_spline_age_sexbatch_transfer.py
    -controls '../data/FA_NIMHANS_ROItable_denoised_controls_final.csv'
    -dirM '/path/to/output_dir/norm_model'
    -dirO 'test_transfer/'
    -age_column age
    -site_column Protocol
    -sex_column sex
    -random_state 413624  
```
One can pick any `random_state` in order to run it 10 different times, each time for a different trained model.

[1]: <https://www.biorxiv.org/content/10.1101/2024.12.15.628527v1>
