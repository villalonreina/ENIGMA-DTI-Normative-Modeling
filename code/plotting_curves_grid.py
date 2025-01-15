# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 2024

@author: Julio Villalón
"""

from vis_utils_julio_agesex import plot_normative_models_multi_predict

import argparse
__author__ = 'Julio Villalón'

parser = argparse.ArgumentParser(description='This runs the predictions for the already trained models.')
parser.add_argument('-model_dir','--model_dir', help='Where the already trained model is.',required=True)
parser.add_argument('-output_dir','--output_dir', help='Where the dummy predicted model will go.',required=True)
parser.add_argument('-metric','--metric', help='What diffusion MRI metric we will use.',required=True)
parser.add_argument('-roi','--roi', help='Name of the ROI to estimate.',required=True)
parser.add_argument('-age_min','--age_min', help='Minimum age for predicitions.',required=True)
parser.add_argument('-age_max','--age_max', help='Maximum age for predicitions.',required=True)
parser.add_argument('-sex_batch','--sex_batch', action="store_true",
                    help='Whether the variable sex was treated as batch variable.',required=False)
args = parser.parse_args()

# Note: The store_true option has a default value of False.
# Whereas, store_false has a default value of True
# So, in order to make sex_batch=True, the flag --sex_batch has to be included
# when calling the function plotting_curves_grid.py. If the --sex_batch flag 
# is not included then sex_batch=False
print("This is the model directory: %s" % args.model_dir)
print("This is the output directory: %s" % args.output_dir)
print("This is the dMRI metric: %s" % args.metric)
print("This is the ROI: %s" % args.roi)
print("This is the minimum age: %s" % args.age_min)
print("This is the maximum age: %s" % args.age_max)
print("Sex handled as a batch?: %s" % args.sex_batch)

metric = args.metric
roi_id = args.roi
age_min = int(args.age_min)
age_max = int(args.age_max)
sex_batch = args.sex_batch

model_1 = metric + '_controls1'
model_2 = metric + '_controls2'
model_3 = metric + '_controls3'
model_4 = metric + '_controls4'
model_5 = metric + '_controls5'
model_6 = metric + '_controls6'
model_7 = metric + '_controls7'
model_8 = metric + '_controls8'
model_9 = metric + '_controls9'
model_10 = metric + '_controls10'

#Predicting one at a time.     
model_dir = args.model_dir + model_1
out_dir = args.output_dir + model_1
# This is for the Nat_Comm runs. We have a total of 37 sites. Plotting the curve for CAMCAN (site 13, with index 0-36)
plot_normative_models_multi_predict(out_dir, model_dir, 37, roi_id, metric, age_min, age_max, sex_batch)

model_dir = args.model_dir + model_2
out_dir = args.output_dir + model_2
# This is for the Nat_Comm runs. We have a total of 37 sites. Plotting the curve for CAMCAN (site 13, with index 0-36)
plot_normative_models_multi_predict(out_dir, model_dir, 37, roi_id, metric, age_min, age_max, sex_batch)

model_dir = args.model_dir + model_3
out_dir = args.output_dir + model_3
# This is for the Nat_Comm runs. We have a total of 37 sites. Plotting the curve for CAMCAN (site 13, with index 0-36)
plot_normative_models_multi_predict(out_dir, model_dir, 37, roi_id, metric, age_min, age_max, sex_batch)

model_dir = args.model_dir + model_4
out_dir = args.output_dir + model_4
# This is for the Nat_Comm runs. We have a total of 37 sites. Plotting the curve for CAMCAN (site 13, with index 0-36)
plot_normative_models_multi_predict(out_dir, model_dir, 37, roi_id, metric, age_min, age_max, sex_batch)

model_dir = args.model_dir + model_5
out_dir = args.output_dir + model_5
# This is for the Nat_Comm runs. We have a total of 37 sites. Plotting the curve for CAMCAN (site 13, with index 0-36)
plot_normative_models_multi_predict(out_dir, model_dir, 37, roi_id, metric, age_min, age_max, sex_batch)

model_dir = args.model_dir + model_6
out_dir = args.output_dir + model_6
# This is for the Nat_Comm runs. We have a total of 37 sites. Plotting the curve for CAMCAN (site 13, with index 0-36)
plot_normative_models_multi_predict(out_dir, model_dir, 37, roi_id, metric, age_min, age_max, sex_batch)

model_dir = args.model_dir + model_7
out_dir = args.output_dir + model_7
# This is for the Nat_Comm runs. We have a total of 37 sites. Plotting the curve for CAMCAN (site 13, with index 0-36)
plot_normative_models_multi_predict(out_dir, model_dir, 37, roi_id, metric, age_min, age_max, sex_batch)

model_dir = args.model_dir + model_8
out_dir = args.output_dir + model_8
# This is for the Nat_Comm runs. We have a total of 37 sites. Plotting the curve for CAMCAN (site 13, with index 0-36)
plot_normative_models_multi_predict(out_dir, model_dir, 37, roi_id, metric, age_min, age_max, sex_batch)

model_dir = args.model_dir + model_9
out_dir = args.output_dir + model_9
# This is for the Nat_Comm runs. We have a total of 37 sites. Plotting the curve for CAMCAN (site 13, with index 0-36)
plot_normative_models_multi_predict(out_dir, model_dir, 37, roi_id, metric, age_min, age_max, sex_batch)

model_dir = args.model_dir + model_10
out_dir = args.output_dir + model_10
# This is for the Nat_Comm runs. We have a total of 37 sites. Plotting the curve for CAMCAN (site 13, with index 0-36)
plot_normative_models_multi_predict(out_dir, model_dir, 37, roi_id, metric, age_min, age_max, sex_batch)

