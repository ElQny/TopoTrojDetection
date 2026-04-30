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
import csv #for logging

import torch
import numpy as np
import pandas as pd
import cv2
from sklearn.metrics import roc_auc_score
from sklearn import preprocessing
from datetime import date
from tqdm import tqdm
from pathlib import Path
import xgboost as xgb
import argparse
import pickle as pkl
import glob
import matplotlib.pyplot as plt
import sys
import importlib

from topological_feature_extractor_img import topo_psf_feature_extract_img
from topological_feature_extractor_pc import topo_psf_feature_extract_pc
from topological_feature_extractor_sphere import topo_psf_feature_extract_sphere

from pointcloud_helper import load_off_file
from run_crossval import run_crossval_xgb, run_crossval_mlp
from plots import plot_topo_features

# Algorithm Configuration
STEP_SIZE:  int = 2 # Stimulation stepsize used in PSF
PATCH_SIZE: int = 2 # Stimulation patch size used in PSF
STIM_LEVEL: int = 5 # Number of stimulation level used in PSF

FILTRATION_METHOD: str='alpha' #vr (Vietoris-Rips) or alpha
N_SAMPLE_NEURONS: int = 1500  # Number of neurons for sampling
USE_EXAMPLE: bool =  False     # Whether clean inputs will be given or not
CORR_METRIC: str = 'distcorr'   # Correlation metric to be used: distcorr, pearson, bc, cos, js
CLASSIFIER: str  = 'xgboost'    # Classifier for the detection , choice = {xgboost, mlp}.

# Experiment Configuration
INPUT_SIZE_IMG: List = [1, 28, 28] #Input images' shape (default to be MNIST)
INPUT_SIZE_PC: List = [23,3,1024] #Input Pointclouds shape (default PCBA)

INPUT_RANGE: List = [0, 255]   # Input image range
TRAIN_TEST_SPLIT: float = 0.8  # Ratio of train to test

# Pointcloud-specific:
NUMBER_OF_POINTS = 2048
BATCH_SIZE = 16
GRANULARITY = 4

# spheres:
CENTER_STEP = 0.2
RADIUS_STEP = 0.05
RADIUS_MIN = 0.05
RADIUS_MAX = 0.3
N_POINTS_TRIGGER = 64

#rounding:
ROUND_DECIMALS = 1


def append_to_csv(filename:str, fieldnames:list, row:dict):
    directory = os.path.dirname(filename)

    if directory:
        os.makedirs(directory, exist_ok=True)
    file_not_found = not (os.path.exists(filename))

    with open(filename, 'a', newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        # if file doesn't exist yet: create with header
        if file_not_found:
            print(f"Creating file {filename}")
            writer.writeheader()
        writer.writerow(row)
        file.flush()


def build_psf_config(args, batch_size, corr_metric, device, granularity, n_neurons, number_of_points, patch_size,
                     step_size, stim_level):
    psf_config = {}
    psf_config['filtration_method'] = FILTRATION_METHOD
    psf_config['step_size'] = STEP_SIZE if step_size is None else step_size
    psf_config['stim_level'] = STIM_LEVEL if stim_level is None else stim_level
    psf_config['patch_size'] = PATCH_SIZE if patch_size is None else patch_size
    psf_config['input_shape'] = INPUT_SIZE_IMG if args.mode == 'img' else INPUT_SIZE_PC
    psf_config['input_range'] = INPUT_RANGE
    psf_config['n_neuron'] = N_SAMPLE_NEURONS if n_neurons is None else n_neurons
    psf_config['corr_method'] = CORR_METRIC if corr_metric is None else corr_metric
    psf_config['device'] = device

    # for Pointclouds:
    if args.mode == 'pc' or args.mode == 'sphere':
        psf_config['number_of_points'] = NUMBER_OF_POINTS if number_of_points is None else number_of_points
        psf_config['granularity'] = GRANULARITY if granularity is None else granularity
        psf_config['batch_size'] = BATCH_SIZE if batch_size is None else batch_size
        psf_config['round_decimals'] = ROUND_DECIMALS

    #for Sphere mode:
    if args.mode == 'sphere':
        psf_config['center_step'] = CENTER_STEP
        psf_config['radius_step'] = RADIUS_STEP
        psf_config['radius_min'] = RADIUS_MIN
        psf_config['radius_max'] = RADIUS_MAX
        psf_config['number_of_points_trigger'] = N_POINTS_TRIGGER

    return psf_config


def load_image_sample(args, model_name, model_train_example_config, root):
    img_c = None
    total_examples = 1  # Default to be a blank image if USE_EXAMPLE=False
    # If use_examples then read in clean input example images
    if USE_EXAMPLE and os.path.exists(model_train_example_config):
        img_c = defaultdict(list)
        example_file = pd.read_csv(model_train_example_config)
        example_file.sample(frac=1)
        n_classes = len(example_file['true_label'].unique())
        for ind in range(example_file.shape[0]):
            if example_file['triggered'].iloc[ind]:
                continue
            c = example_file['true_label'].iloc[ind]
            if not len(img_c[c]):
                img_file = \
                glob.glob(os.path.join(root, model_name, '**', example_file['file'].iloc[ind]), recursive=True)[0]
                img = torch.from_numpy(cv2.imread(img_file, cv2.IMREAD_UNCHANGED)).float()
                img_c[c].append(img.permute(2, 0, 1).unsqueeze(0))
            total_examples = sum([len(img_c[c]) for c in img_c])
            if len(img_c.keys()) == n_classes and total_examples == n_classes:
                break
    return img_c, total_examples


def get_features_for_model(args, model, model_name, model_train_example_config, psf_config, root):
    if args.mode == 'img':
        img_c, total_examples = load_image_sample(args, model_name, model_train_example_config, root)
        fv = topo_psf_feature_extract_img(model, img_c, psf_config)

    elif args.mode == 'pc':
        clean_pc = load_off_file(args.example_pc_path) if args.example_pc_path else None
        fv = topo_psf_feature_extract_pc(model, clean_pc, psf_config)  # random pointclouds
        total_examples = 1

    elif args.mode == 'sphere':
        clean_pc = load_off_file(args.example_pc_path) if args.example_pc_path else None
        fv = topo_psf_feature_extract_sphere(model, clean_pc, psf_config)  # spheres
        total_examples = 1

    else:
        raise ValueError(f"Unknown mode: {args.mode}")
    return fv, total_examples


def calc_log_values(value_range: list, label: str, param_name: str, runs: int, log_path: str):
    fieldnames = ["label", "param_name", "param_value", "index", "seed", "acc", "auc", "ce"]
    filename = os.path.join(log_path, f"{param_name}.csv")  # takes argument log_path from main(args)

    seed = args.seed
    for value in value_range:
        for i in range(runs):
            args.seed = seed + i
            kwargs: dict = {param_name: value}
            acc_test, auc_test, ce_test = main(args, **kwargs)

            append_to_csv(
                filename=filename,
                fieldnames=fieldnames,
                row={
                    "label": label,
                    "param_name": param_name,
                    "param_value": value,
                    "index": i,
                    "seed": args.seed,
                    "acc": acc_test * 100,  # percentiles
                    "auc": auc_test * 100,  # percentiles
                    "ce": ce_test
                }
            )


def main(
        args,
        patch_size=None,
        stim_level=None,
        step_size=None,
        n_neurons = None,
        corr_metric=None,
        batch_size=None,
        granularity=None,
        number_of_points=None
):

    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_ind
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    psf_config = build_psf_config(
        args = args,
        batch_size = batch_size,
        corr_metric =corr_metric,
        device =device,
        granularity= granularity,
        n_neurons= n_neurons,
        number_of_points=number_of_points,
        patch_size=patch_size,
        step_size=step_size,
        stim_level=stim_level)

    root = args.data_root
    model_list = sorted(os.listdir(root))

    if args.mode == 'pc':
        if not args.pc_root:
            raise ValueError("pc_root is required for Pointclouds!")

    if args.mode == 'pc' or args.mode == 'sphere':
        if not args.import_path:
            raise ValueError("import_path is required for Pointclouds!")

        path_for_imports = str(Path(args.import_path).resolve())
        if path_for_imports not in sys.path:
            sys.path.append(path_for_imports)  # imports the structure of the nn-Model from the model file (from where the pointcloud-neural-network was generated)
        importlib.import_module("model")


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

        for root_m, dirnames, filenames in os.walk(os.path.join(root, model_name)):
            for filename in filenames:
                if filename.endswith('.pt.1') or filename.endswith('.pt'): #appended so that .pt is also valid
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
            model_file_path = model_file_path
            model = torch.load(model_file_path).to(device)
        except Exception as e: #expanded so exception is actually printed
            print(f"Model {model_name} .pt file is missing, skip to next model")
            print(f"Exception: {e}")
            continue
        model.eval()

        try:
            model_config = jsonpickle.decode(open(model_config_path, "r").read())
        except:
            print("Model {} config is missing, skip to next model".format(model_config))
            continue

        if gt_file:
            with open(gt_file, "r") as f: #changed this from args.gt_file to gt_file!
                lines = f.readlines()[0]
                gt = int(lines.strip())
        else:
            gt = ('final_triggered_data_n_total' in model_config.keys())
        gt_list.append(gt)


        model_file_path_prefix = '/'.join(model_file_path.split('/')[:-1])
        save_file_path = os.path.join(model_file_path_prefix, 'test_extracted_psf_topo_feature.pkl')

        fv, total_examples = get_features_for_model(args, model, model_name, model_train_example_config, psf_config,
                                                    root)
        with open(save_file_path, 'wb') as f:
            pkl.dump(fv, f)
        f.close()
        fv_list.append(fv)
        # fv_list[i]['psf_feature_pos'] shape: 2 * nExample * fh * fw * nStimLevel * nClasses

    # Plot part:
    all_topo = []
    for fv in fv_list:
        arr = np.array(fv['topo_feature_pos'])
        all_topo.append(arr)

    all_topo = np.concatenate(all_topo, axis=0)
    plot_topo_features(all_topo)

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
        psf_feature=torch.cat([fv_list[i]['psf_feature_pos'].unsqueeze(0) for i in range(len(fv_list))])
        # TOPO feature shape = N*12 where 12 is the total number of topological feature from dim0 and dim1
        topo_feature = torch.cat([fv_list[i]['topo_feature_pos'].unsqueeze(0) for i in range(len(fv_list))])

        topo_feature[np.where(topo_feature==np.Inf)]=1

        if args.mode == 'pc' or args.mode == 'sphere':
            for i in range(len(fv_list)):
                fv_list[i] = fv_list[i]['psf_feature_pos'].unsqueeze(0)
            psf_feature = torch.cat(fv_list)
            n, _, nEx, nCubes, nPerturb, C = psf_feature.shape
            psf_feature_dat = psf_feature.reshape(n, 2, -1, nPerturb, C)
        else:
            n, _, nEx, fnW, fnH, nStim, C = psf_feature.shape
            psf_feature_dat=psf_feature.reshape(n, 2, -1, nStim, C)

        psf_diff_max=(psf_feature_dat.max(dim=3)[0]-psf_feature_dat.min(dim=3)[0]).max(2)[0].view(len(gt_list), -1)
        psf_med_max=psf_feature_dat.median(dim=3)[0].max(2)[0].view(len(gt_list), -1)
        psf_std_max=psf_feature_dat.std(dim=3).max(2)[0].view(len(gt_list), -1)

        if args.mode=='pc' or args.mode == 'sphere': #NEW
            psf_topk_max = psf_feature_dat.topk(k=min(3, nPerturb), dim=3)[0].mean(2).max(2)[0].view(len(gt_list), -1)
        else:
            psf_topk_max=psf_feature_dat.topk(k=min(3, total_examples), dim=3)[0].mean(2).max(2)[0].view(len(gt_list), -1)
        psf_feature_dat=torch.cat([psf_diff_max, psf_med_max, psf_std_max, psf_topk_max], dim=1)

        dat=torch.cat([psf_feature_dat, topo_feature.view(topo_feature.shape[0], -1)], dim=1) #topo and psf features
        # dat = topo_feature.view(topo_feature.shape[0], -1) #only topo features
        dat=preprocessing.scale(dat)
        gt_list=torch.tensor(gt_list)

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
        # debug-Ausgaben:
        print("labels: ", labels)
        print("y_pred before: ", y_pred)

        T, b=best_model_list['threshold']
        y_pred=torch.sigmoid(b*(torch.tensor(y_pred)-T)).numpy()
        print("y_pred afterwards: ", y_pred)

        acc_test = np.sum((y_pred >= 0.5)==labels)/len(y_pred)
        auc_test = roc_auc_score(labels, y_pred)
        ce_test = np.sum(-(labels * np.log(y_pred) + (1 - labels) * np.log(1 - y_pred))) / len(y_pred)
        print("Final Acc {:.3f}% - Final AUC {:.3f} - Fianl CE {:.3f}".format(acc_test * 100, auc_test, ce_test))
        #logger-ausgaben entfernt da csv-logging

    #für Pointclouds/Spheres nicht implementiert
    if CLASSIFIER=='mlp':
        if not args.mode == 'img':
            raise ValueError("Classifier mlp not implemented for pointclouds")
        dat=[]
        for i in range(len(fv_list)):
            psf_fv_pos_i=fv_list[i]['psf_feature_pos']
            _, nEx, fh, fw, nSim, C = psf_fv_pos_i.shape
            psf_fv_pos_i=psf_fv_pos_i.permute(5, 0, 1, 2, 3, 4)
            psf_fv_pos_i=psf_fv_pos_i.reshape(C, -1)
            topo_fv_pos_i=fv_list[i]['topo_feature_pos'].view(nEx, -1)
            dat_pos_i={'psf_fv_pos_i':psf_fv_pos_i, 'topo_fv_pos_i':topo_fv_pos_i}
            dat.append(dat_pos_i)

        N = len(dat)
        n_train = int(TRAIN_TEST_SPLIT * N)
        ind_reshuffle = np.random.choice(list(range(N)), N, replace=False)
        train_ind = ind_reshuffle[:n_train]
        test_ind = ind_reshuffle[n_train:]

        feature_train = [dat[i] for i in train_ind]
        feature_test = [dat[i] for i in test_ind]
        gt_train = [gt_list[i] for i in train_ind]
        gt_test = [gt_list[i] for i in test_ind]

        # Run the training and hyper-parameter searching process
        print('Running hyper-parameter searching and training')
        best_model_list = run_crossval_mlp(feature_train, gt_train)

        # Evaluation
        output_mv=torch.zeros(len(feature_test), 2).cuda()
        for i in range(len(best_model_list['models'])):
            psf_encoder, topo_encoder, cls = best_model_list['models'][i]
            weight = best_model_list['weight'][i]/sum(best_model_list['weight'])

            psf_encoder.eval()
            topo_encoder.eval()
            cls.eval()
            correct = 0
            total = 0
            for j in range(0, max(int((len(feature_test) - 1) / 32), 1)):
                batch = feature_test[(32*j):min(32*(j+1), len(feature_test))] # 32 is the batch size
                embedding_list = []
                for single_input in batch:
                    psf_fv_pos_i=single_input['psf_fv_pos_i'].cuda()
                    topo_fv_pos_i=single_input['topo_fv_pos_i'].cuda()
                    psf_embedding=psf_encoder(psf_fv_pos_i)
                    # Tricks to handle single data point batch. Repeat this data point 5 times and add some Gaussian noise
                    if len(topo_fv_pos_i)==1:
                        topo_fv_pos_i=topo_fv_pos_i.repeat(5, 1)+torch.randn(5, topo_fv_pos_i.shape[1]).cuda()
                    topo_embedding=topo_encoder(topo_fv_pos_i)
                    embedding=torch.cat([psf_embedding.mean(0).flatten(), topo_embedding.mean(0).flatten()])
                    embedding_list.append(embedding)
                embeddings = torch.cat([x.unsqueeze(0) for x in embedding_list])
                output = cls(embeddings)
                output_mv[(32*j):min(32*(j+1), len(feature_test))]+=output*weight

        gt_test=torch.tensor(gt_test)
        output=output_mv
        score=torch.nn.functional.softmax(output, 1).detach().cpu()
        pred = score.argmax(1)
        correct += pred.eq(gt_test).sum().item()
        total += len(gt_test)
        acc_test = correct / total
        auc_test = roc_auc_score(gt_test.detach().cpu().numpy(), score[:, 1].numpy())
        ce_test = -np.mean(np.array(gt_test)*np.log(np.maximum(score[:,1].numpy(), 1e-4))+(1-np.array(gt_test))*np.log(np.maximum(1-score[:,1].numpy(), 1e-4)))


    print("Final Acc {:.3f}% - Final AUC {:.3f} - Fianl CE {:.3f}".format(acc_test*100, auc_test, ce_test))
    # logger-ausgaben entfernt da csv-logging

    return acc_test, auc_test, ce_test


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Extract feature and train trojan detector for synthetic experiment')
    parser.add_argument('--mode', choices=['img', 'pc', 'sphere'], default='img', help='Mode of the Detector') #https://docs.python.org/3/library/argparse.html - on/off flag
    parser.add_argument('--data_root', type=str, help='Root folder that saves the experiment models')
    parser.add_argument('--log_path', type=str, help='Output log save dir', default='./tmp')
    parser.add_argument('--gpu_ind', type=str, help='Indices of GPUs to be used', default='0')
    parser.add_argument('--seed', type=int, help="Experiment random seed", default=123)
    parser.add_argument('--pc_root', type=str, help="Root folder to the pointcloud models")
    parser.add_argument('--import_path', type=str, help="Import path for the pointcloud model")
    parser.add_argument('--example_pc_path', type=str, help='Path to clean pointcloud as OFF file')
    args = parser.parse_args()

    value_range_std = range(10)
    corr_metrics = ["distcorr", "pearson", "bc", "cos", "js"]

    # calc_log_values([1500, 2500, 3500], "N_SAMPLE_NEURONS", "n_neurons", 10, args.log_path)
    # calc_log_values(value_range_std, "STEP_SIZE", "step_size", 10, args.log_path)
    # calc_log_values(value_range_std, "PATCH_SIZE", "patch_size", 10, args.log_path)
    # calc_log_values(value_range_std, "STIM_LEVEL", "stim_level", 10, args.log_path)
    # calc_log_values(corr_metrics, "CORR_METRIC", "corr_metric", 10, args.log_path)

    #NORMAL OUTPUT:
    acc_test, auc_test, ce_test = main(args)
    print(f"ACC: {acc_test}, AUC: {auc_test}, CE: {ce_test}")