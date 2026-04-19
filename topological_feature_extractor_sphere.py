#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import gc
from collections import defaultdict
from typing import List, Tuple, Dict, Optional, Any

import torch
import numpy as np
from ripser import Rips
from scipy import sparse
from scipy.sparse.csr import csr_matrix
import time

from topo_utils import mat_bc_adjacency, parse_arch, feature_collect, sample_act, mat_discorr_adjacency, mat_cos_adjacency, mat_jsdiv_adjacency, mat_pearson_adjacency
from topological_feature_extractor import *
from pointcloud_helper import *


def read_pointcloud_psf_config(psf_config: Dict):
    # reads out parameters from psf_config (dictionary)
    # step, radius_min, radius_max, res, number_of_points
    number_of_points = psf_config['number_of_points']
    granularity = psf_config['granularity']
    batch_size = psf_config['batch_size']
    n_neuron_sample = psf_config['n_neuron']
    method = psf_config['corr_method']
    device = psf_config['device']

    center_step = psf_config['center_step']
    radius_min = psf_config['radius_min']
    radius_max = psf_config['radius_max']
    radius_step = psf_config['radius_step']
    number_of_points_trigger = psf_config['number_of_points_trigger']
    return (
        n_neuron_sample,
        method,
        device,
        number_of_points,
        granularity,
        batch_size,
        center_step,
        radius_min,
        radius_max,
        radius_step,
        number_of_points_trigger)


def generate_perturbed_pointcloud_batch(batch_size, c_idx: int, cubes, device, example_pointcloud, granularity, points_in_cube) -> torch.Tensor:
    print("Generating perturbed pointcloud batch")
    perturbed_pointclouds = []
    for b in range(batch_size):
        temp_perturbed_pc = perturb_points_in_cube(
            pointcloud=example_pointcloud,
            points_in_subcube=points_in_cube,
            cube=cubes[c_idx],
            granularity=granularity,
        )
        perturbed_pointclouds.append(temp_perturbed_pc)

    perturbed_pointclouds = np.array(perturbed_pointclouds)
    tensor = transpose_and_batch_pointclouds_to_tensor(perturbed_pointclouds).to(device)
    return tensor

def generate_activation_vector_matrix(feature_dict_c: Dict) -> torch.Tensor:
    print("Generating activation vector matrix")
    neural_activation_matrix = []
    for k in feature_dict_c:
        #Conv1d is (B, C, N) with B==2 / B==3 etc.!!
        if len(feature_dict_c[k][0].shape) == 2:
            layer_act = [
                feature_dict_c[k][i].max(1)[0].unsqueeze(1)
                for i in range(len(feature_dict_c[k]))
            ]
        else:
            layer_act = [
                feature_dict_c[k][i].unsqueeze(1)
                for i in range(len(feature_dict_c[k]))
            ]

        layer_act = torch.cat(layer_act, dim=1)
        # Standardize the activation layer-wisely
        layer_act = ((layer_act - layer_act.mean(1, keepdim=True))
                                   / (layer_act.std(1, keepdim=True) + 1e-30))
        neural_activation_matrix.append(layer_act)

    neural_activation_matrix = torch.cat(neural_activation_matrix, dim=0)
    return neural_activation_matrix

def build_neural_correlation_matrix(neural_act: torch.Tensor, method:str) -> torch.Tensor:
    print("Building neural correlation matrix")
    if method == 'distcorr':
        neural_pd = mat_discorr_adjacency(neural_act)
    elif method == 'bc':
        neural_act = torch.softmax(neural_act, 1)
        neural_pd = mat_bc_adjacency(neural_act)
    elif method == 'cos':
        neural_pd = mat_cos_adjacency(neural_act)
    elif method == 'pearson':
        neural_pd = mat_pearson_adjacency(neural_act)
    elif method == 'js':
        neural_act = torch.softmax(neural_act, 1)
        neural_pd = mat_jsdiv_adjacency(neural_act)
    else:
        raise Exception(f"Correlation metric {method} isn't implemented !")
    return neural_pd

def build_persist_homology(PD_list, method, model: torch.nn.Module, neural_pd, rips: Rips):
    print("Building persist homology matrix")
    D = 1 - neural_pd.detach().cpu().numpy() \
        if method != 'bc' \
        else -np.log(neural_pd.detach().cpu().numpy() + 1e-6)
    PD_list.append(neural_pd.detach().cpu().numpy())
    if model._get_name == 'ModdedLeNet5Net':
        PH = rips.fit_transform(D, distance_matrix=True)  # directly calling ripser
    else:
        lambdas = getGreedyPerm(D)  # furthest-point-sampling
        D = getApproxSparseDM(lambdas, 0.1, D)  # approx. distance matrix building
        PH = rips.fit_transform(D, distance_matrix=True)  # calling ripser -> faster calculation for larger networks
    return PH


def compute_topological_features(PH):
    print("Computing topological features")
    PH[0] = np.array(PH[0])
    PH[1] = np.array(PH[1])

    PH[0][np.where(PH[0] == np.inf)] = 1
    PH[1][np.where(PH[1] == np.inf)] = 1

    # Compute the topological feature with the persistent diagram
    clean_feature_0 = calc_topo_feature(PH, 0)  # 6 topological features for dimension 0
    clean_feature_1 = calc_topo_feature(PH, 1)  # 6 topological features for dimension 1

    topo_feature = []  # append all these features to topo_feature array -> 12 features
    for k in sorted(list(clean_feature_0)):
        topo_feature.append(clean_feature_0[k])
    for k in sorted(list(clean_feature_1)):
        topo_feature.append(clean_feature_1[k])
    topo_feature = torch.tensor(topo_feature)
    return topo_feature


def topo_psf_feature_extract_sphere(model: torch.nn.Module, example_pointcloud: Dict, psf_config: Dict) -> Dict:
    """
        Combines all above functions as well as helper functions:
        - builds the pointcloud (without any example pointclouds)
        - generates perturbed pointclouds by cube-wise-perturbation
        - uses the DNN previously generated to get activation vectors
        - generates distance matrix from vectors
        - vectors are turned into topological features
    """

    (n_neuron_sample,
    method,
    device,
    number_of_points,
    granularity,
    batch_size,
    center_step,
    radius_min,
    radius_max,
    radius_step,
    number_of_points_trigger) = read_pointcloud_psf_config(psf_config) #touple from function above


    model = model.to(device)
    model.eval() #TODO: is this necessary?

    if example_pointcloud is None:
        clean_pointcloud = create_sample_pointcloud(number_of_points)
    else:
        clean_pointcloud = np.array(example_pointcloud)

    clean_pointcloud = center_and_scale(clean_pointcloud)

    PD_list=[]
    rips = Rips(verbose=False)
    layer_list, _ = parse_arch(model)

    centers = possible_sphere_centers(center_step)

    test_in, radii = generate_radius_batch(
        clean_pointcloud,
        centers[0],
        radius_min,
        radius_max,
        radius_step,
        number_of_points_trigger
    )
    test_in = test_in.to(device)

    test_out = model(test_in[:1])
    if isinstance(test_out, tuple):
        test_out = test_out[0]
    num_classes = int(test_out.shape[1])

    topo_feature_pos = torch.zeros(  #fixed size in zeroes
        len(centers),
        12,
        dtype=torch.float32
    )

    psf_feature_pos = torch.zeros(
        2,  # score + confidence
        1,  # number of examples (1 example pointcloud)
        len(centers), #number of cubes
        len(radii),# perturbations per cube
        num_classes,#number of output classes
        dtype=torch.float32
    )

    for c_idx in range(len(centers)):
        print(f'Center #{c_idx}: {centers[c_idx]}')

        tensor, radii = generate_radius_batch(
            clean_pointcloud,
            centers[c_idx],
            radius_min,
            radius_max,
            radius_step,
            number_of_points_trigger
        )
        tensor = tensor.to(device)

        feature_dict_c, output = feature_collect(model, tensor) #returns hooked activations and model output
        if isinstance(output, tuple):
            output = output[0]

        psf_score = output.detach().cpu()
        psf_conf = torch.softmax(psf_score, dim=1)

        psf_feature_pos[0,0,c_idx, :, :] = psf_score
        psf_feature_pos[1,0,c_idx, :,:] = psf_conf

        neural_act = generate_activation_vector_matrix(feature_dict_c)  # hook-features -> neural activation matrix

        if len(neural_act) > 1.5e3:
            neural_act, sample_n_neurons_list = sample_act(neural_act, layer_list, sample_size=n_neuron_sample)

        neural_pd = build_neural_correlation_matrix(neural_act, method)  # Build neural correlation matrix (depending on correlation method)
        PH = build_persist_homology(PD_list, method, model, neural_pd, rips)   # Distance Matrix generation (D = 1-correlation) -> weights for correlation matrix!
        topo_feature = compute_topological_features(PH) # PH = persistent homology (basically persistence diagram)
        topo_feature_pos[c_idx, :] = topo_feature #overwrite where the feature is in the tensor (so topo_feature_pos stays at same size even with 0es)

    fv = {}
    fv['topo_feature_pos'] = topo_feature_pos
    fv['correlation_matrix'] = np.vstack([x[None, :, :] for x in PD_list]).mean(0)
    fv['psf_feature_pos'] = psf_feature_pos
    return fv
