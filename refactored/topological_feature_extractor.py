#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import gc
from typing import List, Tuple, Dict, Optional, Any
from scipy import sparse
from scipy.sparse.csr import csr_matrix

import numpy as np
import torch

from ripser import Rips
from torch import Tensor
from torch.nn import Module

from pointcloud_helper import *
from topo_utils import *

# Total number of neurons to be sampled
SAMPLE_LIMIT = 3e3


def makeSparseDM(D: np.array, threshold: float)-> np.array:
    """
    Convert a dense matrix to COO format. All values that are below thresh are set to be 0.
    Input args:
        D (np.array): matrix to be converted
        threshold (float): threshold below which value will be set to 0
    Return:
        matrix in compressed sparse column format
    """
    N = D.shape[0]
    [I, J] = np.meshgrid(np.arange(N), np.arange(N))
    I = I[D <= threshold]
    J = J[D <= threshold]
    V = D[D <= threshold]
    return sparse.coo_matrix((V, (I, J)), shape=(N, N)).tocsr()


def getGreedyPerm(D: np.array)-> List:
    """
    A Naive O(N^2) algorithm to do furthest points sampling
    Input args:
        D (np.array):  An NxN distance matrix for points
    Return:
        lamdas (List): list Insertion radii of all points
    """

    N = D.shape[0]
    # By default, takes the first point in the permutation to be the
    # first point in the point cloud, but could be random
    perm = np.zeros(N, dtype=np.int64)
    lambdas = np.zeros(N)
    ds = D[0, :]
    for i in range(1, N):
        idx = np.argmax(ds)
        perm[i] = idx
        lambdas[i] = ds[idx]
        ds = np.minimum(ds, D[idx, :])
    return lambdas[perm]


def getApproxSparseDM(lambdas: List, eps: float, D: np.array)-> csr_matrix:
    """
    Purpose: To return the sparse edge list with the warped distances, sorted by weight.
    Input args:
        lambdas (List): insertion radii for points
        eps (float): epsilon approximation constant
        D (np.array): NxN distance matrix, okay to modify because last time it's used
    Return:
        DSparse (scipy.sparse): A sparse NxN matrix with the reweighted edges
    """
    N = D.shape[0]
    E0 = (1+eps)/eps
    E1 = (1+eps)**2/eps

    # Create initial sparse list candidates (Lemma 6)
    # Search neighborhoods
    nBounds = ((eps**2+3*eps+2)/eps)*lambdas

    # Set all distances outside of search neighborhood to infinity
    D[D > nBounds[:, None]] = np.inf
    [I, J] = np.meshgrid(np.arange(N), np.arange(N))
    idx = I < J
    I = I[(D < np.inf)*(idx == 1)]
    J = J[(D < np.inf)*(idx == 1)]
    D = D[(D < np.inf)*(idx == 1)]

    #Prune sparse list and update warped edge lengths (Algorithm 3 pg. 14)
    minlam = np.minimum(lambdas[I], lambdas[J])
    maxlam = np.maximum(lambdas[I], lambdas[J])

    # Rule out edges between vertices whose balls stop growing before they touch
    # or where one of them would have been deleted.  M stores which of these
    # happens first
    M = np.minimum((E0 + E1)*minlam, E0*(minlam + maxlam))

    t = np.arange(len(I))
    t = t[D <= M]
    (I, J, D) = (I[t], J[t], D[t])
    minlam = minlam[t]
    maxlam = maxlam[t]

    # Now figure out the metric of the edges that are actually added
    t = np.ones(len(I))

    # If cones haven't turned into cylinders, metric is unchanged
    t[D <= 2*minlam*E0] = 0

    # Otherwise, if they meet before the M condition above, the metric is warped
    D[t == 1] = 2.0*(D[t == 1] - minlam[t == 1]*E0) # Multiply by 2 convention
    return sparse.coo_matrix((D, (I, J)), shape=(N, N)).tocsr()


def calc_topo_feature(PH: List, dim: int)-> Dict:
    """
    Compute topological feature from the persistent diagram.
    Input args:
        PH (List) : Persistent diagram
        dim (int) : dimension to be focused on
    Return:
        Dictionary contains topological feature
    """
    pd_dim = PH[dim]
    if dim == 0:
        pd_dim = pd_dim[:-1]
    pd_dim = np.array(pd_dim)
    betti = len(pd_dim)
    ave_persis = sum(pd_dim[:, 1] - pd_dim[:, 0]) / betti if betti > 0 else 0
    ave_midlife = (sum((pd_dim[:, 0] + pd_dim[:, 1]) / 2) / betti) if betti > 0 else 0
    med_midlife = np.median((pd_dim[:, 0] + pd_dim[:, 1]) / 2) if betti > 0 else 0
    max_persis = (pd_dim[:, 1] - pd_dim[:, 0]).max() if betti > 0 else 0
    top_5_persis = np.mean(np.sort(pd_dim[:, 1] - pd_dim[:, 0])[-5:]) if betti > 0 else 0
    topo_feature_dict = {"betti_" + str(dim): betti,
                         "avepersis_" + str(dim): ave_persis,
                         "avemidlife_" + str(dim): ave_midlife,
                         "maxmidlife_" + str(dim): med_midlife,
                         "maxpersis_" + str(dim): max_persis,
                         "toppersis_" + str(dim): top_5_persis}
    return topo_feature_dict


def read_pointcloud_psf_config(psf_config: Dict):
    # reads out parameters from psf_config (dictionary)
    n_neuron_sample = psf_config['n_neuron']
    method = psf_config['corr_method']
    device = psf_config['device']
    number_of_points = psf_config['number_of_points']
    granularity = psf_config['granularity']
    batch_size = psf_config['batch_size']
    return n_neuron_sample, method, device, number_of_points, granularity, batch_size


def generate_perturbed_pointcloud_batch(batch_size, c_idx: int, cubes, device, example_pointcloud, granularity, points_in_cube) -> Tensor:
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
        # todo: Conv1d ist (B, C, N) mit B==2 / B==3 etc.!!
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

def build_persist_homology(PD_list, method, model: Module, neural_pd, rips: Rips):
    print("Building persist homology matrix")
    D = 1 - neural_pd.detach().cpu().numpy() \
        if method != 'bc' \
        else -np.log(neural_pd.detach().cpu().numpy() + 1e-6)
    PD_list.append(neural_pd.detach().cpu().numpy())
    if model._get_name() == 'ModdedLeNet5Net':
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


def topo_psf_feature_extract(model: torch.nn.Module, example_pointcloud: Dict, psf_config: Dict) -> Dict:
    """
        Combines all above functions as well as helper functions:
        - builds the pointcloud (without any example pointclouds)
        - generates perturbed pointclouds by cube-wise-perturbation
        - uses the DNN previously generated to get activation vectors
        - generates distance matrix from vectors
        - vectors are turned into topological features
    """

    n_neuron_sample, method, device, number_of_points, granularity, batch_size = read_pointcloud_psf_config(psf_config)

    model = model.to(device)
    model.eval()

    if example_pointcloud is None:
        example_pointcloud = create_sample_pointcloud(number_of_points)
    example_pointcloud = center_and_scale(example_pointcloud)

    cubes = generate_cubes(granularity)
    sub_pointclouds = choose_sub_pointclouds(
        pointcloud = example_pointcloud,
        granularity=granularity
    )

    # cube-wise perturbation strategy:
    PD_list=[]
    rips = Rips(verbose=False)
    layer_list, _ = parse_arch(model)

    topo_feature_pos = torch.zeros(len(cubes), 12, dtype=torch.float32) #fixed size in zeroes

    for c_idx in range(len(cubes)):
        print("Cube #", c_idx, ":")
        points_in_cube = sub_pointclouds[c_idx]
        if len(points_in_cube) == 0: #skip empty cubes
            continue
        tensor = generate_perturbed_pointcloud_batch(
            batch_size, c_idx, cubes, device, example_pointcloud, granularity,
                                                     points_in_cube)
        feature_dict_c, output = feature_collect(model, tensor) #returns hooked activations and model output

        neural_act = generate_activation_vector_matrix(feature_dict_c)  # hook-features -> neural activation matrix

        if len(neural_act) > 1.5e3:
            neural_act, sample_n_neurons_list = sample_act(neural_act, layer_list, sample_size=n_neuron_sample)

        neural_pd = build_neural_correlation_matrix(neural_act, method)  # Build neural correlation matrix (depending on correlation method)
        PH = build_persist_homology(PD_list, method, model, neural_pd, rips)   # Distance Matrix generation (D = 1-correlation) -> weights for correlation matrix!
        topo_feature = compute_topological_features(PH) # PH = persistent homology (basically persistence diagram)
        topo_feature_pos[c_idx, :] = topo_feature

    fv = {}
    fv['topo_feature_pos'] = topo_feature_pos
    fv['correlation_matrix'] = np.vstack([x[None, :, :] for x in PD_list]).mean(0)
    return fv
