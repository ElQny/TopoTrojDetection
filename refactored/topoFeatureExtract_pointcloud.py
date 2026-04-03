#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import copy
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
from ripser import Rips
from scipy import sparse
from scipy.sparse.csr import csr_matrix

from topo_utils import (
    mat_bc_adjacency,
    parse_arch,
    feature_collect,
    sample_act,
    mat_discorr_adjacency,
    mat_cos_adjacency,
    mat_jsdiv_adjacency,
    mat_pearson_adjacency,
)


# ---------------------------------------------------------------------
# Sparse distance-matrix helpers
# ---------------------------------------------------------------------

def make_sparse_distance_matrix(D: np.ndarray, threshold: float) -> csr_matrix:
    """
    Convert a dense distance matrix to sparse CSR format.
    Only entries <= threshold are kept.
    """
    n = D.shape[0]
    I, J = np.meshgrid(np.arange(n), np.arange(n))
    mask = D <= threshold
    return sparse.coo_matrix((D[mask], (I[mask], J[mask])), shape=(n, n)).tocsr()


def get_greedy_permutation_lambdas(D: np.ndarray) -> np.ndarray:
    """
    Furthest-point sampling insertion radii.

    Parameters
    ----------
    D : np.ndarray
        NxN distance matrix.

    Returns
    -------
    np.ndarray
        Insertion radii ordered by greedy permutation.
    """
    n = D.shape[0]
    perm = np.zeros(n, dtype=np.int64)
    lambdas = np.zeros(n)

    ds = D[0, :]
    for i in range(1, n):
        idx = np.argmax(ds)
        perm[i] = idx
        lambdas[i] = ds[idx]
        ds = np.minimum(ds, D[idx, :])

    return lambdas[perm]


def get_approx_sparse_distance_matrix(
    lambdas: np.ndarray,
    eps: float,
    D: np.ndarray,
) -> csr_matrix:
    """
    Build approximate sparse filtration matrix following the original logic.
    """
    n = D.shape[0]
    e0 = (1 + eps) / eps
    e1 = (1 + eps) ** 2 / eps

    # Neighborhood bounds
    n_bounds = ((eps ** 2 + 3 * eps + 2) / eps) * lambdas

    # Keep only edges within search neighborhood
    D = D.copy()
    D[D > n_bounds[:, None]] = np.inf

    I, J = np.meshgrid(np.arange(n), np.arange(n))
    upper_triangle_mask = I < J
    finite_mask = (D < np.inf) & upper_triangle_mask

    I = I[finite_mask]
    J = J[finite_mask]
    edge_distances = D[finite_mask]

    # Prune candidate edges
    minlam = np.minimum(lambdas[I], lambdas[J])
    maxlam = np.maximum(lambdas[I], lambdas[J])

    M = np.minimum((e0 + e1) * minlam, e0 * (minlam + maxlam))
    keep = edge_distances <= M

    I = I[keep]
    J = J[keep]
    edge_distances = edge_distances[keep]
    minlam = minlam[keep]

    # Warp distances if needed
    warped_mask = edge_distances > 2 * minlam * e0
    edge_distances[warped_mask] = 2.0 * (
        edge_distances[warped_mask] - minlam[warped_mask] * e0
    )

    return sparse.coo_matrix((edge_distances, (I, J)), shape=(n, n)).tocsr()


# ---------------------------------------------------------------------
# Persistent-homology feature helpers
# ---------------------------------------------------------------------

def calc_topo_feature(PH: List[np.ndarray], dim: int) -> Dict[str, float]:
    """
    Compute six summary statistics from a persistence diagram in one dimension.
    """
    pd_dim = PH[dim]

    # Remove the final infinite H0 class, consistent with original code
    if dim == 0:
        pd_dim = pd_dim[:-1]

    pd_dim = np.array(pd_dim)
    betti = len(pd_dim)

    if betti == 0:
        return {
            f"betti_{dim}": 0,
            f"avepersis_{dim}": 0,
            f"avemidlife_{dim}": 0,
            f"maxmidlife_{dim}": 0,
            f"maxpersis_{dim}": 0,
            f"toppersis_{dim}": 0,
        }

    persistence = pd_dim[:, 1] - pd_dim[:, 0]
    midlife = (pd_dim[:, 0] + pd_dim[:, 1]) / 2

    return {
        f"betti_{dim}": betti,
        f"avepersis_{dim}": persistence.mean(),
        f"avemidlife_{dim}": midlife.mean(),
        f"maxmidlife_{dim}": np.median(midlife),   # preserved original behavior
        f"maxpersis_{dim}": persistence.max(),
        f"toppersis_{dim}": np.mean(np.sort(persistence)[-5:]),
    }


def persistence_features_to_tensor(PH: List[np.ndarray]) -> torch.Tensor:
    """
    Convert PH diagrams (dim 0 and dim 1) into a 12-dimensional feature tensor.
    """
    feat0 = calc_topo_feature(PH, 0)
    feat1 = calc_topo_feature(PH, 1)

    values = []
    for key in sorted(feat0.keys()):
        values.append(feat0[key])
    for key in sorted(feat1.keys()):
        values.append(feat1[key])

    return torch.tensor(values, dtype=torch.float32)


def sanitize_persistence_diagram(PH: List[np.ndarray]) -> List[np.ndarray]:
    """
    Convert diagrams to numpy arrays and replace inf with 1, preserving original behavior.
    """
    PH[0] = np.array(PH[0])
    PH[1] = np.array(PH[1])
    PH[0][np.isinf(PH[0])] = 1
    PH[1][np.isinf(PH[1])] = 1
    return PH


# ---------------------------------------------------------------------
# PSF / perturbation helpers
# ---------------------------------------------------------------------

def get_default_examples(input_shape: List[int]) -> Dict[int, List[torch.Tensor]]:
    """
    Create a default example dictionary with one blank input.
    """
    example_dict = defaultdict(list)
    example_dict[0].append(torch.zeros(input_shape).unsqueeze(0))
    return example_dict


def get_num_classes(model: torch.nn.Module, example_dict: Dict[int, List[torch.Tensor]], device) -> int:
    """
    Infer number of classes from one forward pass.
    """
    test_input = example_dict[0][0].to(device)
    return int(model(test_input).shape[1])


def get_stimulation_sequence(input_value_range: List[float], stim_level: int) -> np.ndarray:
    """
    Evenly spaced stimulation values.
    """
    return np.linspace(input_value_range[0], input_value_range[1], stim_level)


def get_feature_map_shape(input_shape: List[int], patch_size: int, step_size: int) -> Tuple[int, int]:
    """
    Number of patch positions along height and width.
    """
    feature_map_h = len(range(0, input_shape[1] - patch_size + 1, step_size))
    feature_map_w = len(range(0, input_shape[2] - patch_size + 1, step_size))
    return feature_map_h, feature_map_w


def initialize_feature_tensors(
    num_examples: int,
    feature_map_h: int,
    feature_map_w: int,
    num_stim_levels: int,
    num_classes: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Allocate output tensors.
    """
    psf_feature_pos = torch.zeros(
        2,                      # score / confidence
        num_examples,
        feature_map_h,
        feature_map_w,
        num_stim_levels,
        num_classes,
    )

    topo_feature_pos = torch.zeros(
        num_examples,
        feature_map_h * feature_map_w,
        12,                     # 6 features for H0 + 6 features for H1
    )

    return psf_feature_pos, topo_feature_pos


def build_perturbed_inputs(
    input_example: torch.Tensor,
    stim_seq: np.ndarray,
    pos_h: int,
    pos_w: int,
    patch_size: int,
    input_shape: List[int],
) -> torch.Tensor:
    """
    Create a batch of inputs where one spatial patch is stimulated across multiple values.
    """
    perturbed = input_example.repeat(len(stim_seq), 1, 1, 1)

    for idx, stim_value in enumerate(stim_seq):
        perturbed[
            idx,
            :,
            pos_w:min(pos_w + patch_size, input_shape[1]),
            pos_h:min(pos_h + patch_size, input_shape[2]),  # likely correct width bound
        ] = stim_value

    return perturbed


def run_model_in_batches(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    device,
    batch_size: int = 8,
) -> Tuple[Dict, torch.Tensor]:
    """
    Forward perturbed inputs through the model and collect logits + intermediate features.
    """
    logits_list = []
    feature_dict = None

    if len(inputs) < 32:
        feature_dict, output = feature_collect(model, inputs.to(device))
        logits_list.append(output.detach().cpu())
    else:
        for start in range(0, len(inputs), batch_size):
            batch = inputs[start:start + batch_size].to(device)
            feature_dict, output = feature_collect(model, batch)
            logits_list.append(output.detach().cpu())

    logits = torch.cat(logits_list, dim=0)
    return feature_dict, logits


def extract_layerwise_neural_activations(feature_dict: Dict) -> torch.Tensor:
    """
    Convert collected intermediate activations into a 2D neuron-activation matrix.

    Output shape:
        [num_neurons, num_stimulation_examples]
    """
    neural_act = []

    for key in feature_dict:
        layer_inputs = feature_dict[key]

        # Conv-like input: [C, H, W] per sample after indexing
        if len(layer_inputs[0].shape) == 3:
            layer_act = [
                layer_inputs[i].max(1)[0].max(1)[0].unsqueeze(1)
                for i in range(len(layer_inputs))
            ]
        else:
            layer_act = [
                layer_inputs[i].unsqueeze(1)
                for i in range(len(layer_inputs))
            ]

        layer_act = torch.cat(layer_act, dim=1)

        # Standardize each neuron across stimulation samples
        layer_act = (
            layer_act - layer_act.mean(1, keepdim=True)
        ) / (
            layer_act.std(1, keepdim=True) + 1e-30
        )

        neural_act.append(layer_act)

    return torch.cat(neural_act, dim=0)


def maybe_sample_neurons(
    neural_act: torch.Tensor,
    model: torch.nn.Module,
    n_neuron_sample: int,
) -> Tuple[torch.Tensor, Optional[List[int]]]:
    """
    Sample neurons if the activation matrix is too large.
    """
    sampled_counts = None
    layer_list = parse_arch(model)

    if len(neural_act) > 1.5e3:
        neural_act, sampled_counts = sample_act(
            neural_act,
            layer_list,
            sample_size=n_neuron_sample,
        )

    return neural_act, sampled_counts


def compute_neural_correlation_matrix(neural_act: torch.Tensor, method: str) -> torch.Tensor:
    """
    Build neuron-neuron similarity / correlation matrix.
    """
    if method == "distcorr":
        return mat_discorr_adjacency(neural_act)

    if method == "bc":
        neural_act = torch.softmax(neural_act, dim=1)
        return mat_bc_adjacency(neural_act)

    if method == "cos":
        return mat_cos_adjacency(neural_act)

    if method == "pearson":
        return mat_pearson_adjacency(neural_act)

    if method == "js":
        neural_act = torch.softmax(neural_act, dim=1)
        return mat_jsdiv_adjacency(neural_act)

    raise ValueError(f"Correlation metric '{method}' is not implemented.")


def correlation_to_distance_matrix(neural_pd: torch.Tensor, method: str) -> np.ndarray:
    """
    Convert correlation/similarity matrix to distance matrix for PH.
    """
    pd_np = neural_pd.detach().cpu().numpy()

    if method == "bc":
        return -np.log(pd_np + 1e-6)

    return 1 - pd_np


def should_use_dense_filtration(model: torch.nn.Module) -> bool:
    """
    Preserve the original intent: use dense filtration for smaller LeNet-like model.
    """
    # Original code used model._get_name without calling it; that likely never worked as intended.
    return model._get_name() == "ModdedLeNet5Net"


def compute_persistence_diagram(
    distance_matrix: np.ndarray,
    model: torch.nn.Module,
    rips: Rips,
) -> List[np.ndarray]:
    """
    Compute persistence diagram, optionally using sparse approximation.
    """
    if should_use_dense_filtration(model):
        PH = rips.fit_transform(distance_matrix, distance_matrix=True)
    else:
        lambdas = get_greedy_permutation_lambdas(distance_matrix)
        sparse_dm = get_approx_sparse_distance_matrix(lambdas, eps=0.1, D=distance_matrix)
        PH = rips.fit_transform(sparse_dm, distance_matrix=True)

    return sanitize_persistence_diagram(PH)


def process_single_patch(
    model: torch.nn.Module,
    input_example: torch.Tensor,
    stim_seq: np.ndarray,
    pos_h: int,
    pos_w: int,
    patch_size: int,
    input_shape: List[int],
    n_neuron_sample: int,
    method: str,
    device,
    rips: Rips,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, np.ndarray]:
    perturbed_inputs = build_perturbed_inputs(
        input_example=input_example,
        stim_seq=stim_seq,
        pos_h=pos_h,
        pos_w=pos_w,
        patch_size=patch_size,
        input_shape=input_shape,
    )

    feature_dict, logits = run_model_in_batches(model, perturbed_inputs, device)
    confidence = torch.softmax(logits, dim=1)

    neural_act = extract_layerwise_neural_activations(feature_dict)
    neural_act, _ = maybe_sample_neurons(neural_act, model, n_neuron_sample)

    neural_pd = compute_neural_correlation_matrix(neural_act, method)
    distance_matrix = correlation_to_distance_matrix(neural_pd, method)

    PH = compute_persistence_diagram(distance_matrix, model, rips)
    topo_features = persistence_features_to_tensor(PH)

    return logits, confidence, topo_features, neural_pd.detach().cpu().numpy()


def topo_psf_feature_extract(
    model: torch.nn.Module,
    example_dict: Dict,
    psf_config: Dict,
) -> Dict:
    step_size = psf_config["step_size"]
    stim_level = psf_config["stim_level"]
    patch_size = psf_config["patch_size"]
    input_shape = psf_config["input_shape"]
    input_value_range = psf_config["input_range"]
    n_neuron_sample = psf_config["n_neuron"]
    method = psf_config["corr_method"]
    device = psf_config["device"]

    if not example_dict:
        example_dict = get_default_examples(input_shape)

    model = model.to(device)
    num_classes = get_num_classes(model, example_dict, device)
    stim_seq = get_stimulation_sequence(input_value_range, stim_level)
    feature_map_h, feature_map_w = get_feature_map_shape(input_shape, patch_size, step_size)

    psf_feature_pos, topo_feature_pos = initialize_feature_tensors(
        num_examples=len(example_dict.keys()),
        feature_map_h=feature_map_h,
        feature_map_w=feature_map_w,
        num_stim_levels=len(stim_seq),
        num_classes=num_classes,
    )

    correlation_matrices = []
    rips = Rips(verbose=False)

    for class_idx in example_dict:
        input_example = copy.deepcopy(example_dict[class_idx][0])

        feature_w_pos = 0
        for pos_w in range(0, input_shape[1] - patch_size + 1, step_size):
            feature_h_pos = 0
            for pos_h in range(0, input_shape[2] - patch_size + 1, step_size):

                logits, confidence, topo_features, corr_matrix = process_single_patch(
                    model=model,
                    input_example=input_example,
                    stim_seq=stim_seq,
                    pos_h=pos_h,
                    pos_w=pos_w,
                    patch_size=patch_size,
                    input_shape=input_shape,
                    n_neuron_sample=n_neuron_sample,
                    method=method,
                    device=device,
                    rips=rips,
                )

                psf_feature_pos[0, class_idx, feature_w_pos, feature_h_pos] = logits
                psf_feature_pos[1, class_idx, feature_w_pos, feature_h_pos] = confidence

                flat_pos = feature_w_pos * feature_map_w + feature_h_pos
                topo_feature_pos[class_idx, flat_pos, :] = topo_features

                correlation_matrices.append(corr_matrix)
                feature_h_pos += 1

            feature_w_pos += 1

    return {
        "psf_feature_pos": psf_feature_pos,
        "topo_feature_pos": topo_feature_pos,
        "correlation_matrix": np.vstack([x[None, :, :] for x in correlation_matrices]).mean(0),
    }