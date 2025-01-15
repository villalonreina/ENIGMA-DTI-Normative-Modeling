<img width="1461" alt="ENIGMA-DTI_Normative_Modeling_logos" src="https://github.com/user-attachments/assets/c63a98e1-57be-4d16-a85c-126857cd40d2" />




This repository contains the code to run normative modeling with hierarchical Bayesian regression (HBR) using the [PCNtoolkit](https://pcntoolkit.readthedocs.io/en/latest/) on white matter summary metrics derived from diffusion tensor imaging (DTI), i.e., fractional anisotropy (FA), mean diffusivity (MD), radial diffusivity (RD) and axial diffusivity (AD). A detailed description of how to process the diffusion MRI data with the ENIGMA-DTI protocol to extract white matter derivatives can be found here: [ENIGMA-DTI Protocols](https://enigma.ini.usc.edu/protocols/dti-protocols/). This code was used to develop and deploy the normative models presented in this paper: [Lifespan Normative Modeling of Brain Microstructure][1]

## System Requirements
The code in this repo requires the following:
- [python](https://www.python.org/) (>= 3.9)
- [PCNtoolkit](https://pcntoolkit.readthedocs.io/en/latest/) (>= 0.29)

The instructions to install the PCNtoolkit can be found here: [PCNtoolkit installation](https://github.com/amarquand/PCNtoolkit).

## Demo of model development
Here we provide a smaller version of the dataset used in the paper. One can run the training code on Ipython:






## Demo of model adaptation

```
run nm_hbr_NIMHANS_spline_age_sexbatch_transfer.py
    -controls '/Volumes/four_d/jvillalo/NormativeModel/data_sheets/Denoising_data_Nature/FA_NIMHANS_ROItable_denoised_controls_final.csv'
    -dirM '/Volumes/four_d/jvillalo/NormativeModel/NormModel_age_sexbatch_spline_NatComms_v29_stand/FA_controls1/'
    -dirO '/Users/julio/Documents/IGC/HBR_paper/test_transfer/'
    -age_column age
    -site_column Protocol
    -sex_column sex
    -random_state 413624  
```

[1]: <https://www.biorxiv.org/content/10.1101/2024.12.15.628527v1>
