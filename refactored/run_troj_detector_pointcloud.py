#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Date    : 2021-12-17 12:00:00
# @Author  : Songzhu Zheng (imzszhahahaha@gmail.com)
# @Link    : https://songzhu-academic-site.netlify.app/

import os
import random
import json
import jsonpickle
from collections import defaultdict
from typing import List

import torch
import numpy as np
import pandas as pd
import cv2
from sklearn.metrics import roc_auc_score
from sklearn import preprocessing
import xgboost as xgb
import argparse
import pickle as pkl
from datetime import date
from tqdm import tqdm
import glob

from topological_feature_extractor import topo_psf_feature_extract
from run_crossval import run_crossval_xgb, run_crossval_mlp

# Algorithm Configuration
STEP_SIZE:  int = 2 # Stimulation stepsize used in PSF
PATCH_SIZE: int = 2 # Stimulation patch size used in PSF
STIM_LEVEL: int = 4 # Number of stimulation level used in PSF
N_SAMPLE_NEURONS: int = 1.5e3  # Number of neurons for sampling
USE_EXAMPLE: bool =  False     # Whether clean inputs will be given or not
CORR_METRIC: str = 'distcorr'   # Correlation metric to be used
CLASSIFIER: str  = 'xgboost'    # Classifier for the detection , choice = {xgboost, mlp}.
# Experiment Configuration
INPUT_SIZE: List = [23,3,1024] # Input images' shape (default to be MNIST)
INPUT_RANGE: List = [0, 255]   # Input image range
TRAIN_TEST_SPLIT: float = 0.8  # Ratio of train to test

PC_ROOT: str = '/home/lisa/Desktop/Bac2_Program/modelnet40'
IMPORT_PATH: str = '/home/lisa/Desktop/Bac2_Program/PCBA'

import sys
from pathlib import Path
sys.path.append(str(Path(IMPORT_PATH).resolve())) #imports the structure of the nn-Model from the model file (from where the pointcloud-neural-network was generated)
import model

def main(args):
 #Seeding:
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_ind
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# init Dictionary psf_config with input values
    psf_config = {}
    psf_config['step_size'] = STEP_SIZE
    psf_config['stim_level'] = STIM_LEVEL
    psf_config['patch_size'] = PATCH_SIZE
    psf_config['input_shape'] = INPUT_SIZE
    psf_config['input_range'] = INPUT_RANGE
    psf_config['n_neuron'] = N_SAMPLE_NEURONS
    psf_config['corr_method'] = CORR_METRIC
    psf_config['pc_root'] = PC_ROOT
    psf_config['device'] = device

    root = args.data_root
    model_list = sorted(os.listdir(root))

    # --------------------------------- Step I: Feature Extraction ---------------------------------
    print(">>> Step I: Feature Extraction <<<")
    gt_list = []
    fv_list = []

    for j in tqdm(range(len(model_list)), ncols=50, ascii=True):

        model_name = model_list[j]
        model_file_path = []
        model_config_path = []
        model_train_example_config = None
        gt_file = None

        #sets paths to files / folders
        for root_m, dirnames, filenames in os.walk(os.path.join(root, model_name)):
            for filename in filenames:
                if filename.endswith('.pt.1') or filename.endswith('.pt'):
                    model_file_path = os.path.join(root_m, filename)
                if filename.endswith('gt.txt'):
                    gt_file = os.path.join(root_m, filename)
                if filename.endswith('.json'):
                    model_config_path = os.path.join(root_m, filename)
                if filename.endswith('experiment_train.csv'):
                    model_train_example_config = os.path.join(root_m, filename)
            if len(model_file_path) and len(model_config_path) and model_train_example_config:
                break

        try:
            print(f"Modelfile path {model_file_path}")
            model = torch.load(model_file_path) #load model for evaluation
            print(type(model))
            test = model.to(device)
        except Exception as e:
            print("Model {} .pt file is missing, skip to next model".format(model_name))
            print(e)
            continue
        model.eval()

        try:
            model_config = jsonpickle.decode(open(model_config_path, "r").read())
        except:
            print("Model {} config is missing, skip to next model".format(model_config))
            continue

        if gt_file: #if gt exists
            with open(gt_file, "r") as f: #changed this from args.gt_file -> gt_file (since gt_file was initialized above)
                lines = f.readlines()[0]
                gt = int(lines.strip()) #read out what the gt is (trojanized or not?)
        else:
            gt = ('final_triggered_data_n_total' in model_config.keys()) #else check the json for filan_triggered_data_n_total
        gt_list.append(gt) #if that exists -> it is trojanized, else it is not trojanized -> append this to gt_list

        img_c = None #!! Picture for pixel-wise peturbation is blank (default)
        total_examples = 1 # Default to be a blank image if USE_EXAMPLE=False
        # If use_examples then read in clean input example images

        model_file_path_prefix = '/'.join(model_file_path.split('/')[:-1])
        save_file_path = os.path.join(model_file_path_prefix, 'test_extracted_psf_topo_feature.pkl')
        fv = topo_psf_feature_extract(model, img_c, psf_config) #extract the features from the model -> TODO: change this since we are working with different features now!
        with open(save_file_path, 'wb') as f:
            pkl.dump(fv, f)
        f.close()
        fv_list.append(fv)
        # fv_list[i]['psf_feature_pos'] shape: 2 * nExample * fh * fw * nStimLevel * nClasses

    # --------------------------------- Step II: Train Classifier ---------------------------------
    print(">>> Step II: Train Classifier <<<")
    if CLASSIFIER=='xgboost':

        # PSF feature shape = N*2*m*w*h*L*C
        #   n: number of models
        #   2: logits and confidence
        #   m: number of input images
        #   w: width of the feature map
        #   h: height of the feature map
        #   L: number of stimulation levels
        #   C: number of classes
        # psf_feature=torch.cat([fv_list[i]['psf_feature_pos'].unsqueeze(0) for i in range(len(fv_list))])
        # TOPO feature shape = N*12 where 12 is the total number of topological feature from dim0 and dim1
        topo_feature = torch.cat([fv_list[i]['topo_feature_pos'].unsqueeze(0) for i in range(len(fv_list))]) #collects all features from PSF

        # topo_feature[np.where(topo_feature==np.Inf)]=1
        # n, _, nEx, fnW, fnH, nStim, C = psf_feature.shape
        # psf_feature_dat=psf_feature.reshape(n, 2, -1, nStim, C)
        # psf_diff_max=(psf_feature_dat.max(dim=3)[0]-psf_feature_dat.min(dim=3)[0]).max(2)[0].view(len(gt_list), -1)
        # psf_med_max=psf_feature_dat.median(dim=3)[0].max(2)[0].view(len(gt_list), -1)
        # psf_std_max=psf_feature_dat.std(dim=3).max(2)[0].view(len(gt_list), -1)
        # psf_topk_max=psf_feature_dat.topk(k=min(3, total_examples), dim=3)[0].mean(2).max(2)[0].view(len(gt_list), -1)
        # psf_feature_dat=torch.cat([psf_diff_max, psf_med_max, psf_std_max, psf_topk_max], dim=1)

        # dat=torch.cat([psf_feature_dat, topo_feature.view(topo_feature.shape[0], -1)], dim=1)
        dat = topo_feature.view(topo_feature.shape[0], -1) #collects all topological features
        dat=preprocessing.scale(dat)
        gt_list=torch.tensor(gt_list)

# do train-test Split!
        N = len(gt_list)
        n_train = int(TRAIN_TEST_SPLIT * N)
        ind_reshuffle = np.random.choice(list(range(N)), N, replace=False)
        train_ind = ind_reshuffle[:n_train]
        test_ind = ind_reshuffle[n_train:]

        feature_train, feature_test = dat[train_ind], dat[test_ind]
        gt_train, gt_test = gt_list[train_ind], gt_list[test_ind]

        # Run the training and hyper-parameter searching process
        print('Running hyper-parameter searching and training')
        best_model_list = run_crossval_xgb(np.array(feature_train), np.array(gt_train))

        feature = feature_test
        labels = np.array(gt_test)
        dtest = xgb.DMatrix(np.array(feature), label=labels)
        y_pred = 0
        for i in range(len(best_model_list['models'])):
            best_bst=best_model_list['models'][i]
            weight=best_model_list['weight'][i]/sum(best_model_list['weight'])
            y_pred += best_bst.predict(dtest)*weight

        y_pred = y_pred / len(best_model_list)
        T, b=best_model_list['threshold']
        y_pred=torch.sigmoid(b*(torch.tensor(y_pred)-T)).numpy()
        acc_test = np.sum((y_pred >= 0.5)==labels)/len(y_pred)
        auc_test = roc_auc_score(labels, y_pred)
        ce_test = np.sum(-(labels * np.log(y_pred) + (1 - labels) * np.log(1 - y_pred))) / len(y_pred)


    logger_name=date.today().strftime("%d-%m-%Y")+'_synthetic_'+"-".join([str(x) for x in psf_config['input_shape']])
    logger_file=os.path.join(args.log_path, logger_name)
    if not os.path.exists(args.log_path):
        os.mkdir(args.log_path)
    logger=open(logger_file, 'w')
    print("Final Acc {:.3f}% - Final AUC {:.3f} - Fianl CE {:.3f}".format(acc_test*100, auc_test, ce_test))
    logger.write("Final Acc {:.3f}% - Final AUC {:.3f} - Fianl CE {:.3f}".format(acc_test*100, auc_test, ce_test))
    logger.flush()
    logger.close()

    return acc_test, auc_test, ce_test

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Extract feature and train trojan detector for synthetic experiment')
    parser.add_argument('--data_root', type=str, help='Root folder that saves the experiment models')
    parser.add_argument('--log_path', type=str, help='Output log save dir', default='./tmp')
    parser.add_argument('--gpu_ind', type=str, help='Indices of GPUs to be used', default='0')
    parser.add_argument('--seed', type=int, help="Experiment random seed", default=123)
    args = parser.parse_args()

    exp_logfile=date.today().strftime("%d-%m-%Y")+f'{CORR_METRIC}_{CLASSIFIER}_{N_SAMPLE_NEURONS}.json'
    exp_logfile=os.path.join(args.log_path, exp_logfile)
    exp_config={}
    for k, v in args._get_kwargs():
        exp_config[k]=v

    acc_test, auc_test, ce_test = main(args)

    with open(exp_logfile, 'w') as f:
        json.dump(exp_config, f, sort_keys=False, indent=4)
    f.close()
