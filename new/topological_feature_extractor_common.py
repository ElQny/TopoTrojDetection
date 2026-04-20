import gudhi
import numpy as np
import torch

from typing import List, Dict
from ripser import Rips
from scipy import sparse
from scipy.sparse.csr import csr_matrix
from sklearn.manifold import MDS

from plots import *

from topo_utils import (
    mat_bc_adjacency,
    mat_discorr_adjacency,
    mat_cos_adjacency,
    mat_jsdiv_adjacency,
    mat_pearson_adjacency,
)

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


def generate_activation_vector_matrix(feature_dict_c: Dict):
    """
    takes input from the hooks for each layer
    Linear: (B, F) with B=Batch, F=Features
    Conv1d: (B, C, N) with C=Channel, N=Position (1D-Position)
    Conv2d: (B, C, H, W) with H=Height, W=Width (2D-Position)

    maximizes to reduce to (B,C) then standardize over batch-samples
    """
    print("Generating activation vector matrix")
    neural_activation_matrix = []
    layer_ids = []
    layer_names = []

    for key in feature_dict_c: #feature_dict = Dictionary, accessing via key!
        batch = feature_dict_c[key] #vectors for same batch
        activation_vector_len = len(batch.shape)

        if activation_vector_len == 2: #linear
            layer_act = [
                act.unsqueeze(1)# (C) -> (C,1)
                for act in batch #act = (C)
            ]
        elif activation_vector_len == 3: #Conv1d
            layer_act = [
                act.max(dim=1)[0].unsqueeze(1) #(C,N) -> (C) -> (C,1)
                for act in batch # act = (C, N)
            ]
        elif activation_vector_len == 4: #Conv2d
            layer_act = [
                act.max(dim=1)[0].max(dim=1)[0].unsqueeze(1) #(C,H,W) -> (C,H) -> (C) -> (C,1)
                for act in batch # act = (C, H, W)
            ]
        else:
            raise ValueError(f'Activation Tensor has unknown shape: {batch.shape}')
        layer_act = torch.cat(layer_act, dim=1) # layer_act= (C, 1) -> (C, B)
        # Standardize the activation layer-wisely
        layer_act = ((layer_act - layer_act.mean(1, keepdim=True))
                                   / (layer_act.std(1, keepdim=True) + 1e-30))
        neural_activation_matrix.append(layer_act)

        layer_id, layer_name = key
        for c in range(layer_act.shape[0]): #all channels
            layer_ids.append(layer_id)
            layer_names.append(layer_name)

    neural_activation_matrix = torch.cat(neural_activation_matrix, dim=0)
    return neural_activation_matrix, np.array(layer_ids), np.array(layer_names)

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

def build_persist_homology_vr(PD_list, method, model: torch.nn.Module, neural_pd, rips: Rips):
    print("Building persist homology matrix using ripser")
    #build Distance Matrix:
    D = 1 - neural_pd.detach().cpu().numpy() \
        if method != 'bc' \
        else -np.log(neural_pd.detach().cpu().numpy() + 1e-6) #for Bhattacharyya
    PD_list.append(neural_pd.detach().cpu().numpy())
    if model._get_name == 'ModdedLeNet5Net':
        PH = rips.fit_transform(D, distance_matrix=True)  # directly calling ripser
    else:
        lambdas = getGreedyPerm(D)  # furthest-point-sampling
        D = getApproxSparseDM(lambdas, 0.1, D)  # approx. distance matrix building
        PH = rips.fit_transform(D, distance_matrix=True)  # calling ripser -> faster calculation for larger networks
    return PH

def topo_feature_from_corr_matrix(
        PD_list,
        method,
        neural_pd,
        model,
        rips,
        filtration_method: str = 'vr',
        layer_ids = None,
        layer_names = None
        ):
    # ALPHA path:
    if filtration_method == 'alpha':
        PH = build_persist_homology_alpha(PD_list, method, neural_pd, 3, 1, layer_ids, layer_names)
        PH = adapt_topological_features_alpha(PH)
    # VR path:
    elif filtration_method == 'vr':
        PH = build_persist_homology_vr(PD_list, method, model, neural_pd, rips)   # Distance Matrix generation (D = 1-correlation) -> weights for correlation matrix!

    topo_feature = compute_topological_features(PH)  # PH = persistent homology (basically persistence diagram)
    return topo_feature

"""
Alpha-Complex instead of Vietoris-Rips
"""
def build_persist_homology_alpha(
        PD_list,
        method,
        neural_pd,
        space_for_mapping: int = 3,
        random_state: int = 0,
        layer_ids = None,
        layer_names = None):
    """
    uses alpha-complex to calculate ph, input is space_for_mapping as the
    dimensionality of a lower-dimensional euklidian space where distance
    matrix can be projected to;
    """
    print("Building persist homology matrix using alpha complex")
    D = 1 - neural_pd.detach().cpu().numpy() \
        if method != 'bc' \
        else -np.log(neural_pd.detach().cpu().numpy() + 1e-6)  # for Bhattacharyya
    PD_list.append(neural_pd.detach().cpu().numpy())

    #maps Data to lower Dimensional space
    mds = MDS(n_components=space_for_mapping,
              dissimilarity="precomputed", #for distance matrix
              random_state=random_state)
    #check symmetry:
    print("max asymmetry:", np.abs(D - D.T).max())

    #D not entirely symmetric -> needs to be symmetric for mds!
    D = np.asarray(D, dtype=np.float64)
    D = 0.5 * (D + D.T)
    np.fill_diagonal(D, 0.0)
    D = np.maximum(D, 0.0)

    points = mds.fit_transform(D) #needs symmetric array!

    print("Plotting the Pointcloud...")
    if space_for_mapping == 3:
        title = f'MDS pointcloud in Euklidian space'
        # plot_pointcloud(points, title) #for plots
        plot_pointcloud_layers(pointcloud=points, layers = layer_ids, layer_names = layer_names, title=title)

    alpha_complex = gudhi.AlphaComplex(points = points)
    stree = alpha_complex.create_simplex_tree()
    PH = stree.persistence()

    plot_persist_diagram(PH)

    return PH #(dimension, (birth, death))

def adapt_topological_features_alpha(PH): #PH=(dimension, (birth, death)) -> [PH0, PH1]
    """
    Adapter for compute_topological_features
    takes PH from build_persist_homology_alpha and prepares PH for compute_topological_features(PH)
    """
    PH0 = []
    PH1 = []
    PH_arr = [PH0, PH1]

    for dimension, life_death in PH:
        if dimension == 0:
            PH_arr[0].append(life_death)
        elif dimension == 1:
            PH_arr[1].append(life_death)
    return PH_arr

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

