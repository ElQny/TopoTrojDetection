#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Date    : 2021-12-17 12:00:00
# @Author  : Songzhu Zheng (imzszhahahaha@gmail.com)
# @Link    : https://songzhu-academic-site.netlify.app/

import copy
from collections import defaultdict
from typing import List, Dict, Any

import torch
import numpy as np
from ripser import Rips
from scipy import sparse
from scipy.sparse.csr import csr_matrix
import time

import os, glob
import h5py
from torch.utils.data import Dataset, DataLoader

from topo_utils import mat_bc_adjacency, parse_arch, feature_collect, sample_act, mat_discorr_adjacency, mat_cos_adjacency, mat_jsdiv_adjacency, mat_pearson_adjacency


class H5PointCloudDataset(Dataset):
    def __init__(self, h5_dir, num_points=1024):
        self.files = sorted(glob.glob(os.path.join(h5_dir, "*.h5")))
        self.num_points = num_points

        self.points = []
        self.labels = []
        for fp in self.files:
            with h5py.File(fp, "r") as f:
                data = f["data"][:]   # typically [M,N,3]
                label = f["label"][:] # typically [M,1] or [M]
            self.points.append(data)
            self.labels.append(label)

        self.points = np.concatenate(self.points, axis=0)
        self.labels = np.concatenate(self.labels, axis=0).reshape(-1)

        # optionally truncate/ensure num_points
        if self.points.shape[1] != self.num_points:
            self.points = self.points[:, :self.num_points, :]

    def __len__(self):
        return self.points.shape[0]

    def __getitem__(self, idx):
        pts = torch.from_numpy(self.points[idx]).float()   # [N,3]
        lbl = int(self.labels[idx])
        return pts, lbl

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

def psf_feature_extract(model: torch.nn.Module, example_dict: Dict, psf_config: Dict) -> Dict:
    """
    Extract PSF features (logits + confidence) from a given torch model.
    Input args:
        model (torch.nn.Module). Target model.
        example_dict (Dict). Optional. Dictionary contains clean input examples. If None then all blank images are used.
    Return:
        fv (Dict). Dictionary contains extracted features
    """
    step_size=psf_config['step_size']
    stim_level=psf_config['stim_level']
    patch_size=psf_config['patch_size']
    input_shape=psf_config['input_shape']
    input_valuerange=psf_config['input_range']
    device=psf_config['device']

    # If true input examples are not given, use all blank images instead
    if not example_dict:
        example_dict=defaultdict(list)
        example_dict[0].append(torch.zeros(input_shape).unsqueeze(0))

    model=model.to(device)
    test_input=example_dict[0][0].to(device)
    num_classes=int(model(test_input).shape[1])

    stim_seq=np.linspace(input_valuerange[0], input_valuerange[1], stim_level)
    # 2 represent score and conf
    feature_map_h=len(range(0, input_shape[1]-patch_size+1, step_size))
    feature_map_w=len(range(0, input_shape[2]-patch_size+1, step_size))
    # PSF feature dim : 2*m*h*w*L*C
    #  2: logits and confidence
    #  m: numebr of input examples
    #  h: feature map height
    #  w: feature map width
    #  L: number of stimulation levels
    #  C: number of classes
    psf_feature_pos=torch.zeros(
        2,
        len(example_dict.keys()),
        feature_map_h, feature_map_w,
        len(stim_seq), num_classes)

    progress=0
    # For each class input examples, scan through pixels with step_size and modify corresponding pixel with different stimulation level.
    # Forward all these modified images to the network and collect output logits and confidence
    for c in example_dict:
        input_eg=copy.deepcopy(example_dict[c][0])
        feature_w_pos=0
        for pos_w in range(0, input_shape[1]-patch_size+1, step_size):
            feature_h_pos = 0
            for pos_h in range(0, input_shape[2]-patch_size+1, step_size):
                t0=time.time()
                count=0
                prob_input=input_eg.repeat(len(stim_seq),1,1,1)
                for i in stim_seq:
                    prob_input[count,:,
                               int(pos_w):min(int(pos_w+patch_size), input_shape[1]),
                               int(pos_h):min(int(pos_h+patch_size), input_shape[1])]=i
                    count+=1
                pred=[]
                batch_size=8 if len(prob_input)>=32 else 1
                if batch_size==1:
                    prob_input=prob_input.to(device)
                    feature_dict_c, output = feature_collect(model, prob_input)
                    pred.append(output.detach().cpu())
                else:
                    for b in range(int(len(prob_input)/batch_size)):
                        prob_input_batch=prob_input[(8*b):min(8*(b+1), len(prob_input))].to(device)
                        feature_dict_c, output = feature_collect(model, prob_input_batch)
                        pred.append(output.detach().cpu())
                pred=torch.cat(pred)
                psf_score=pred
                psf_conf=torch.nn.functional.softmax(psf_score, 1)

                psf_feature_pos[0, c, feature_w_pos, feature_h_pos]=psf_score
                psf_feature_pos[1, c, feature_w_pos, feature_h_pos]=psf_conf

                feature_h_pos+=1
            feature_w_pos+=1

    fv={}
    fv['psf_feature_pos']=psf_feature_pos
    return fv


def topo_feature_extract(model: torch.nn.Module, example_dict: Dict, topo_config: Dict) -> Dict:  # KEEP for point clouds: main entry point; signature OK (example_dict becomes point-cloud examples)
    """
    Extract topological features from a given torch model.  # KEEP: docstring still valid (topo features), but “blank images” wording would be inaccurate for point clouds
    Input args:
        model (torch.nn.Module). Target model.  # KEEP: you still need the model
        example_dict (Dict). Optional. Dictionary contains clean input examples. If None then all blank images are used.  # CHANGE: for point clouds it’s “blank point clouds” or remove fallback
    Return:
        fv (Dict). Dictionary contains extracted features  # KEEP: still returns features
    """
    input_shape=topo_config['input_shape']  # CHANGE: not as (C,H,W); only needed if you still create a default input or validate shapes
    n_neuron_sample=topo_config['n_neuron']  # KEEP: still needed to subsample neurons for tractable neuron–neuron graph size
    method=topo_config['corr_method']  # KEEP: still needed to define neuron–neuron similarity metric -> distance matrix for Ripser
    device=topo_config['device']  # KEEP: still needed to run model and move tensors

    # If true input examples are not given, use all blank images instead  # CHANGE: comment refers to images; point clouds would need a different fallback (or remove fallback)
    if not example_dict:  # OPTIONAL for point clouds: only needed if you want the function to run without provided samples (usually you do provide samples)
        example_dict=defaultdict(list)  # OPTIONAL: only needed for fallback construction
        example_dict[0].append(torch.zeros(input_shape).unsqueeze(0))  # CHANGE/NOT NEEDED: creates blank IMAGE shaped by input_shape; for point clouds you'd create [1,N,d] or drop fallback

    model=model.to(device)  # KEEP: model must be on device
    test_input=example_dict[0][0].to(device)  # KEEP if you still want a “probe” forward; otherwise OPTIONAL


#TODO: Dataloader to provide the 32 Point Clouds per batch!!
    # pointclouds that were used for generation of nns
    # loader = ...

    #TODO: currently hardcoded, needs to be handled as input
    # what this does: batch_size ist die Menge an Pointclouds die in der aktuellen Batch miteinander verglichen werden
    # xyz gibt an, welche Dimension die Pointclouds selbst haben
    # 1024 gibt an, wie viele Punkte die Pointclouds haben
    # Effektiv werden hier die 32 Pointclouds in einen 32-dimensionalen Raum transferiert, wo ihre einzelnen Positionen miteinander verglichen werden um sog. activation vectors zu bauen.
    # diese Vektoren (auf deren Basis die Distanz der Neuronen voneinander berechnet werden können) werden dann genutzt, um innerhalb des Raumes die Distanzen zu berechnen
    # daraus wird dann Vietoris-Rips gemacht
    batch_size = 32
    no_batches = 1 #ToDO: increase
    num_workers = 4
    num_points = 1024
    pin_memory = torch.cuda.is_available()

    pc_root = topo_config['pc_root'] #points to the root directory where all the point clouds lie
    dataset = H5PointCloudDataset(h5_dir=pc_root, num_points=num_points)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=num_workers)

    # 12 is the number of topological features (including dim1 and dim2 features)  # KEEP: still 12D summary (H0+H1) regardless of input modality
    # after transpose: [B, 3, N] meaning Batchsize (standard: 32), xyz coordinates (3) and num_points(standard from train_attacked: 1024)
    topo_feature_pos=torch.zeros(  # KEEP conceptually, but CHANGE shape logic: for point clouds you likely want [n_examples, n_batches_or_1, 12] not feature_map_h*feature_map_w
        len(example_dict.keys()),  # KEEP if you preserve example_dict keyed by class/label; otherwise CHANGE to number of groups/batches
        no_batches,  # CHANGE: depends on image scan positions; for point clouds set to 1 or number of batches/perturbations
        12  # KEEP
    )

    PD_list=[]  # KEEP if you want correlation_matrix output; otherwise OPTIONAL
    rips = Rips(verbose=False)  # KEEP: persistent homology backend

    # For each class input examples, scan through pixels with step_size and modify corresponding pixel with different stimulation level.  # CHANGE: image-specific description; for point clouds you won't scan pixels
    # Forward all these modified images to the network and collect output logits and confidence  # CHANGE: PSF/image-specific; topo uses activations, not stored logits
    for c in example_dict:  # KEEP: iterate over provided example groups (classes) if you keep this structure
        for batch_idx, (prob_input, labels) in enumerate(loader):
            if batch_idx >= no_batches: break
            prob_input = prob_input.transpose(2, 1)  # [32, 3, N]

            # Forward all these modified images to the network and collect output logits and confidence  # CHANGE: still need forward pass, but not “images/logits/confidence”
            # (Topology-only: we still need feature_dict_c from feature_collect, but we do not store PSF outputs.)  # KEEP: conceptually correct (still need feature_dict_c)
            if batch_size==1:  # KEEP if you keep minibatching
                prob_input=prob_input.to(device)  # KEEP: move batch to device
                feature_dict_c, output = feature_collect(model, prob_input)  # KEEP: required; this produces internal activations for topology
            else:  # KEEP if you keep minibatching
                for b in range(int(len(prob_input)/batch_size)):  # KEEP if you keep minibatching
                    prob_input_batch=prob_input[(batch_size*b):min(batch_size*(b+1), len(prob_input))].to(device)  # CHANGE/BUG: hard-coded 8; should be batch_size; independent of point clouds but will matter if batch_size != 8
                    feature_dict_c, output = feature_collect(model, prob_input_batch)  # KEEP: required

            # Extract intermediate activating vectors  # KEEP: this is core for topology
            neural_act = []  # KEEP
            for k in feature_dict_c:  # KEEP
                if len(feature_dict_c[k][0].shape)==3:  # KEEP but MAY NEED CHANGE: many point-cloud layers produce 3D tensors but axis semantics differ
                    layer_act = [pool_pointcloud_to_channel_vector(feature_dict_c[k][i]).unsqueeze(1)
                                 for i in range(len(feature_dict_c[k]))]  # now takes into consideration multiple tensor dimensions
                else:  # KEEP
                    layer_act = [feature_dict_c[k][i].unsqueeze(1) for i in range(len(feature_dict_c[k]))]  # KEEP if non-3D activations
                layer_act=torch.cat(layer_act, dim=1)  # KEEP: stacks activations across stim/batch dimension used by feature_collect
                # Standardize the activation layer-wisely  # KEEP: normalizes per-neuron across samples/perturbations
                layer_act=(layer_act-layer_act.mean(1, keepdim=True))/(layer_act.std(1, keepdim=True)+1e-30)  # KEEP
                neural_act.append(layer_act)  # KEEP
            neural_act=torch.cat(neural_act)  # KEEP
            layer_list=parse_arch(model)  # KEEP: needed for layer-wise neuron sampling
            sample_n_neurons_list=None  # OPTIONAL: unused afterwards; can be removed
            if len(neural_act)>1.5e3:  # KEEP conceptually (subsample if too many neurons), threshold is heuristic
                neural_act, sample_n_neurons_list=sample_act(neural_act, layer_list, sample_size=n_neuron_sample)  # KEEP: controls runtime/memory

            # Build neural correlation matrix  # KEEP: core step to build graph/metric for PH
            if method=='distcorr':  # KEEP
                neural_pd=mat_discorr_adjacency(neural_act)  # KEEP
            elif method=='bc':  # KEEP
                neural_act=torch.softmax(neural_act, 1)  # KEEP if bc/js require nonnegative distributions; may need reconsideration depending on activation meaning
                neural_pd=mat_bc_adjacency(neural_act)  # KEEP
            elif method=='cos':  # KEEP
                neural_pd=mat_cos_adjacency(neural_act)  # KEEP
            elif method=='pearson':  # KEEP
                neural_pd=mat_pearson_adjacency(neural_act)  # KEEP
            elif method=='js':  # KEEP
                neural_act=torch.softmax(neural_act, 1)  # KEEP (same note as bc)
                neural_pd=mat_jsdiv_adjacency(neural_act)  # KEEP
            else:  # KEEP
                raise Exception(f"Correlation metrics {method} doesn't implemented !")  # KEEP
            D=1-neural_pd.detach().cpu().numpy() if method!='bc' else -np.log(neural_pd.detach().cpu().numpy()+1e-6)  # KEEP: converts similarity -> distance matrix for Ripser
            PD_list.append(neural_pd.detach().cpu().numpy())  # OPTIONAL: only needed for correlation_matrix output

            # Approaximate sparse filtration to further save some computation  # KEEP: performance optimization for PH on large neuron sets
            if model._get_name=='ModdedLeNet5Net':  # OPTIONAL/LIKELY CHANGE: model name check is image-model specific and probably wrong (method vs attribute)
                PH=rips.fit_transform(D, distance_matrix=True)  # KEEP: valid PH computation
            else:  # KEEP
                lambdas=getGreedyPerm(D)  # KEEP: sparse Rips approximation prep
                D = getApproxSparseDM(lambdas, 0.1, D)  # KEEP: produces sparse distance matrix approximation
                PH=rips.fit_transform(D, distance_matrix=True)  # KEEP

            PH[0]=np.array(PH[0])  # KEEP: normalize type for downstream processing
            PH[1]=np.array(PH[1])  # KEEP
            PH[0][np.where(PH[0]==np.inf)]=1  # KEEP/OPTIONAL: replaces inf deaths; pragmatic for summary stats
            PH[1][np.where(PH[1]==np.inf)]=1  # KEEP/OPTIONAL
            # Compute the topological feature with the persistent diagram  # KEEP: core feature extraction
            clean_feature_0=calc_topo_feature(PH, 0)  # KEEP: H0 summaries
            clean_feature_1=calc_topo_feature(PH, 1)  # KEEP: H1 summaries
            topo_feature=[]  # KEEP
            for k in sorted(list(clean_feature_0)):  # KEEP: stable ordering
                topo_feature.append(clean_feature_0[k])  # KEEP
            for k in sorted(list(clean_feature_1)):  # KEEP
                topo_feature.append(clean_feature_1[k])  # KEEP
            topo_feature=torch.tensor(topo_feature)  # KEEP: tensorize
            topo_feature_pos[c, batch_idx, :]=topo_feature  # indexing depends on image scan position; for point clouds use a batch/perturbation index (often 0)


    fv={}  # KEEP: output dict
    fv['topo_feature_pos']=topo_feature_pos  # KEEP: main output
    fv['correlation_matrix']=np.vstack([x[None, :, :] for x in PD_list]).mean(0)  # OPTIONAL: keep if you want this diagnostic; else remove PD_list accumulation and this line
    return fv  # KEEP


def pool_pointcloud_to_channel_vector(A: torch.Tensor) -> torch.Tensor:
    # Reduce 3d activations (Point-Net-Dimensions: [B, C, N]) over token dimensions
    if A.dim() == 3: #Point-Net: [B,C,N] -> 3d
        B, C, N = A.shape
        pooled = A.max(dim=2)[0] # [B,C] -> 2d
        if B == 1:
            pooled = pooled[0] # [C] -> 1d
        else:
            pooled = pooled.mean(dim=0) # [C] -> 1d
        return pooled

    if A.dim() == 2: # [C,N] -> 2d
        d0, d1 = A.shape
        if d0 <= 256:
            pooled = A.mean(dim=0) # [C] -> 1d
        else:
            pooled = A.max(dim=1)[0]#[C] -> 1d
        return pooled

    if A.dim() == 1: #only [C] -> 1d
        return A # -> 1d

    raise ValueError(f"Shape cannot be reduced to channel vector: {A.shape}") # -> 4D+


def topo_psf_feature_extract(model: torch.nn.Module, example_dict: Dict, topo_config: Dict) -> Dict:
    """
    Extract topological features from a given torch model.
    Input args:
        model (torch.nn.Module). Target model.
        example_dict (Dict). Optional. Dictionary contains clean input examples. If None then all blank images are used.
    Return:
        fv (Dict). Dictionary contains extracted features
    """
    fv = {}

    # --- PSF part -----------------------------------------------------------
    # psf_fv = psf_feature_extract(model, example_dict, psf_config)
    # fv.update(psf_fv)
    # -----------------------------------------------------------------------

    # --- Topological part ---------------------------------------------------
    topo_fv = topo_feature_extract(model, example_dict, topo_config)
    fv.update(topo_fv)
    # -----------------------------------------------------------------------

    return fv