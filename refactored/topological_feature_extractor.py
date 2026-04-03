#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import gc
from typing import List, Tuple, Dict, Optional

import numpy as np
import torch

from ripser import Rips
from pointcloud_helper import create_sample_pointcloud_for_tensor, gen_tensor_from_pointcloud

# Total number of neurons to be sampled
SAMPLE_LIMIT = 3e3


# ---------------------------------------------------------------------
# Image preprocessing
# ---------------------------------------------------------------------

# # TODO: replace
# def compute_center_crop_offsets_for_224(
#     img_shape: Tuple[int, int, int]
# ) -> Tuple[int, int]:
#     """
#     Compute the center-crop offsets used by the original preprocessing logic.
#     """
#     h, w, c = img_shape
#     dx = int((w - 224) / 2)
#     dy = int((w - 224) / 2)
#     return dx, dy
#
#
# # TODO: replace
# def crop_image_to_224_center(img: np.ndarray, dx: int, dy: int) -> np.ndarray:
#     """
#     Crop an image to a 224x224 center crop using precomputed offsets.
#     """
#     return img[dy:dy + 224, dx:dx + 224, :]
#
#
# # TODO: replace
# def convert_hwc_to_chw(img: np.ndarray) -> np.ndarray:
#     """
#     Convert image layout from HWC to CHW.
#     """
#     return np.transpose(img, (2, 0, 1))
#
#
# # TODO: replace
# def add_batch_dimension(img: np.ndarray) -> np.ndarray:
#     """
#     Convert CHW tensor-like array to NCHW by adding batch dimension.
#     """
#     return np.expand_dims(img, 0)
#
#
# # TODO: replace
# def subtract_image_minimum(img: np.ndarray) -> np.ndarray:
#     """
#     Shift image values so the minimum becomes zero.
#     """
#     return img - np.min(img)
#
#
# # TODO: replace
# def divide_by_image_maximum(img: np.ndarray) -> np.ndarray:
#     """
#     Normalize image by its maximum value.
#     """
#     return img / np.max(img)
#
#
# # TODO: replace
# def convert_numpy_image_to_float_tensor(img: np.ndarray) -> torch.FloatTensor:
#     """
#     Convert numpy array to float tensor.
#     """
#     return torch.FloatTensor(img)
#
#
# # TODO: replace
# def img_std(img):
#     """
#     Reshape and rescale the input images to fit the model.
#     """
#     dx, dy = compute_center_crop_offsets_for_224(img.shape)
#     img = crop_image_to_224_center(img, dx, dy)
#     img = convert_hwc_to_chw(img)
#     img = add_batch_dimension(img)
#     img = subtract_image_minimum(img)
#     img = divide_by_image_maximum(img)
#     batch_data = convert_numpy_image_to_float_tensor(img)
#     return batch_data


# ---------------------------------------------------------------------
# Model architecture parsing
# ---------------------------------------------------------------------

# can stay
def module_has_children(module: torch.nn.Module) -> bool:
    """
    Return True if a module contains child modules.
    """
    return bool(module._modules)


def is_supported_feature_module(module: torch.nn.Module) -> bool:
    """
    Return True if the module is one of the feature-bearing modules used here.
    """
    return isinstance(module, torch.nn.Conv1d) or isinstance(module, torch.nn.Linear)


# can stay
def prefix_submodule_names(parent_name: str, child_names: List[str]) -> List[str]:
    """
    Prefix recursively extracted layer names with the current parent module name.
    """
    return [parent_name + "_" + name for name in child_names]


def parse_arch(model: torch.tensor) -> Tuple[List, List]:
    """
    Parse a input model to extact layer-wise (Conv1d or Linear) module and corresponding module name.

    Input args:
        model (torch.nn.Module): A torch network

    Return:
        layer_list (List): A list contain all Conv1d and Linear module from shallow to deep
        layer_k (List): A list contain names of extracted modules in layer_list
    """
    layer_list = []
    layer_k = []

    for k in model._modules:
        module = model._modules[k]

        if module_has_children(module):
            sub_layer_list, sub_layer_k = parse_arch(module)
            layer_list += sub_layer_list
            layer_k += prefix_submodule_names(k, sub_layer_k)
        elif is_supported_feature_module(module):
            layer_list.append(module)
            layer_k.append(module._get_name())

    return layer_list, layer_k


# ---------------------------------------------------------------------
# Forward-hook feature collection
# ---------------------------------------------------------------------

# can stay
def extract_hook_input_tensor(f_in) -> torch.Tensor:
    """
    Extract the tensor input seen by a forward hook, matching the original logic.
    """
    if isinstance(f_in, torch.Tensor):
        return f_in.detach().cpu()
    return f_in[0].detach().cpu()


# can stay
def register_feature_hooks(module_list: List[torch.nn.Module], hook_fn) -> List:
    """
    Register a forward hook on every module in module_list.
    """
    return [module.register_forward_hook(hook=hook_fn) for module in module_list]


# can stay
def remove_feature_hooks(handle_list: List) -> None:
    """
    Remove all registered forward hooks.
    """
    for handle in handle_list:
        handle.remove()


# can stay
def map_layer_outputs_to_feature_dict(
    outs: List[torch.Tensor],
    module_k: List[str],
) -> Dict:
    """
    Convert collected hook outputs into the original feature_dict structure.
    """
    feature_dict = {}
    for layer_ind in range(len(module_k)):
        feature_dict[(layer_ind, module_k[layer_ind])] = outs[layer_ind]
    return feature_dict


# can stay
def feature_collect(model: torch.tensor, pointclouds: torch.tensor) -> Tuple[Dict, torch.tensor]:
    """
    Helper function to collection intermediate output of a model for given inputs.

    Input args:
        model (torch.nn.Module): A torch network
        pointclouds (torch.tensor): A valid pointcloud torch.tensor

    Return:
        feature_dict (dict): A dictionary contain all intermediate output tensor whose key is the (layer depth, module name)
        output (torch.tensor): final output of model
    """
    outs = []

    def feature_hook(module, f_in, f_out):
        outs.append(extract_hook_input_tensor(f_in))

    module_list, module_k = parse_arch(model)
    handle_list = register_feature_hooks(module_list, feature_hook)
    output = model(pointclouds)
    feature_dict = map_layer_outputs_to_feature_dict(outs, module_k)
    remove_feature_hooks(handle_list)

    return feature_dict, output


# ---------------------------------------------------------------------
# Neuron-count utilities for sampling / block processing
# ---------------------------------------------------------------------
#TODO might have to change to Out_features / out_channels
# reason: could be that the trojanization is only visible after transformation thorugh layer!!

# TODO: replace/extend with out_features??
def extract_conv_input_counts(layer_modules: List[torch.nn.Module]) -> List[int]:
    """
    Extract in_channels for convolutional layers.
    """
    return [x.in_channels for x in layer_modules if hasattr(x, "in_channels")]

# replace with out_features??
def extract_linear_input_counts(layer_modules: List[torch.nn.Module]) -> List[int]:
    """
    Extract in_features for linear layers.
    """
    return [x.in_features for x in layer_modules if hasattr(x, "in_features")]


def combine_layer_neuron_counts(conv_counts: List[int], linear_counts: List[int]) -> List[int]:
    """
    Combine convolutional and linear neuron/filter counts.
    """
    return conv_counts + linear_counts


def compute_cumulative_neuron_boundaries(neuron_counts: List[int]) -> List[int]:
    """
    Convert per-layer neuron counts to cumulative boundaries with a leading zero.
    """
    return [0] + list(np.cumsum(neuron_counts))


# TODO: replace/extend with out_features??
# rename...
def extract_original_layer_neuron_counts(layer_list: List) -> List[int]:
    """
    Reproduce the original layer-neuron extraction logic from layer_list[0].
    """
    conv_nfilters_list = extract_conv_input_counts(layer_list[0])
    linear_nneurons_list = extract_linear_input_counts(layer_list[0])
    return combine_layer_neuron_counts(conv_nfilters_list, linear_nneurons_list)


def compute_stratified_sample_counts(
    neuron_counts: List[int],
    sample_size: int,
) -> List[int]:
    """
    Compute how many neurons to sample from each layer proportionally.
    """
    total_neurons = sum(neuron_counts)
    return [int(sample_size * x / total_neurons) for x in neuron_counts]


def sample_indices_within_layer_boundaries(
    neuron_boundaries: List[int],
    layer_sample_num: List[int],
) -> List[np.ndarray]:
    """
    Sample neuron indices independently within each layer boundary interval.
    """
    return [
        np.random.choice(
            range(neuron_boundaries[i], neuron_boundaries[i + 1]),
            layer_sample_num[i],
            replace=False,
        )
        for i in range(len(neuron_boundaries) - 1)
        if layer_sample_num[i]
    ]


def concatenate_sampled_indices(sample_ind: List[np.ndarray]) -> np.ndarray:
    """
    Concatenate sampled neuron indices into one flat index vector.
    """
    return np.concatenate(sample_ind)


def compute_sampled_neuron_counts_per_layer(sample_ind: List[np.ndarray]) -> List[int]:
    """
    Compute how many neurons were sampled from each retained layer.
    """
    return [len(x) for x in sample_ind]


def select_sampled_neural_activations(
    neural_act: torch.Tensor,
    sample_ind: np.ndarray,
) -> torch.Tensor:
    """
    Select sampled neuron rows from the activation matrix.
    """
    return neural_act[sample_ind]


# TODO: check if used out_features if still works
def sample_act(
    neural_act: torch.tensor,
    layer_list: List,
    sample_size: int,
) -> Tuple[torch.tensor, List]:
    """
    Stratified sampling certain number of neurons' output given all activating vector of a model.

    Input args:
        neural_act (torch.tensor): n*d tensor. n is the total number of neurons and d is number of record (input sample size)
        layer_list (List): a list contain Conv2d or Linear module of a network. it is the return of parse_arch
        sample_size (int): Interger that specifies the number of neurons to be sampled
    """
    n_neurons_list = extract_original_layer_neuron_counts(layer_list)
    layer_sample_num = compute_stratified_sample_counts(n_neurons_list, sample_size)
    neuron_boundaries = compute_cumulative_neuron_boundaries(n_neurons_list)
    sample_ind = sample_indices_within_layer_boundaries(neuron_boundaries, layer_sample_num)
    sample_n_neurons_list = compute_sampled_neuron_counts_per_layer(sample_ind)
    sample_ind = concatenate_sampled_indices(sample_ind)

    return select_sampled_neural_activations(neural_act, sample_ind), sample_n_neurons_list


# ---------------------------------------------------------------------
# Block pooling over layer-wise correlation blocks
# ---------------------------------------------------------------------

# TODO: replace/extend
def resolve_neuron_boundaries_for_process_pd(
    layer_list: List,
    sample_n_neurons_list: Optional[List] = None,
) -> List[int]:
    """
    Resolve the neuron boundaries used for blockwise pooling of the correlation matrix.
    """
    if not sample_n_neurons_list:
        n_neurons_list = extract_original_layer_neuron_counts(layer_list)
        return compute_cumulative_neuron_boundaries(n_neurons_list)
    return [0] + list(np.cumsum(sample_n_neurons_list))


def initialize_block_pooled_pd_matrix(num_layers: int) -> np.ndarray:
    """
    Create the output matrix for pooled inter-layer statistics.
    """
    return np.zeros([num_layers, num_layers])


def set_identity_block_value(maxpool_pd: np.ndarray, i: int, j: int) -> None:
    """
    Set the diagonal self-layer similarity to 1.
    """
    maxpool_pd[i, j] = 1


def extract_interlayer_block(
    pd: torch.Tensor,
    n_neurons_list: List[int],
    i: int,
    j: int,
):
    """
    Extract the block corresponding to interactions between layer i and layer j.
    """
    return pd[
        n_neurons_list[i]:n_neurons_list[i + 1],
        n_neurons_list[j]:n_neurons_list[j + 1],
    ]


def flatten_block_values(block) -> np.ndarray:
    """
    Flatten a block to one dimension.
    """
    return block.flatten()


def compute_top_fraction_count(length: int, fraction: float = 0.4) -> int:
    """
    Compute how many entries belong to the top fraction used by the original logic.
    """
    return int(fraction * length)


def select_top_fraction_indices(block: np.ndarray, fraction: float = 0.4) -> np.ndarray:
    """
    Select the indices of the top fraction of values from a flattened block.
    """
    k = compute_top_fraction_count(len(block), fraction)
    return np.argpartition(block.flatten(), -k)[-k:]


def compute_mean_of_selected_entries(block: np.ndarray, indices: np.ndarray) -> float:
    """
    Compute the mean of selected flattened block entries.
    """
    return block[indices].mean()


def write_symmetric_block_value(
    maxpool_pd: np.ndarray,
    i: int,
    j: int,
    value: float,
) -> None:
    """
    Write a pooled inter-layer value symmetrically.
    """
    maxpool_pd[i, j] = value
    maxpool_pd[j, i] = value


# TODO: replace/extend
def process_pd(
    pd: torch.tensor,
    layer_list: List,
    sample_n_neurons_list: List = None,
) -> torch.tensor:
    """
    Pool the neuron-level pairwise matrix into a layer-level matrix.
    """
    n_neurons_list = resolve_neuron_boundaries_for_process_pd(
        layer_list, sample_n_neurons_list
    )
    maxpool_pd = initialize_block_pooled_pd_matrix(len(n_neurons_list) - 1)

    for i in range(len(n_neurons_list) - 1):
        for j in range(i, len(n_neurons_list) - 1):
            if i == j:
                set_identity_block_value(maxpool_pd, i, j)
            else:
                block = extract_interlayer_block(pd, n_neurons_list, i, j)
                block = flatten_block_values(block)
                per_ind = select_top_fraction_indices(block, fraction=0.4)
                value = compute_mean_of_selected_entries(block, per_ind)
                write_symmetric_block_value(maxpool_pd, i, j, value)

    return maxpool_pd


# ---------------------------------------------------------------------
# Distance-correlation helper math
# ---------------------------------------------------------------------

def resolve_default_second_matrix(
    X: torch.Tensor,
    Y: Optional[torch.Tensor],
) -> torch.Tensor:
    """
    Use X as Y when Y is not provided, matching the original behavior.
    """
    if Y is None:
        return X
    return Y


def compute_pairwise_feature_distances(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    """
    Compute pairwise distances between the feature dimensions of row vectors in X and Y.
    """
    return torch.cdist(X.unsqueeze(2), Y.unsqueeze(2), p=2)


def compute_row_mean(matrix: torch.Tensor) -> torch.Tensor:
    """
    Compute mean over axis 1.
    """
    return matrix.mean(axis=1)


def compute_column_mean(matrix: torch.Tensor) -> torch.Tensor:
    """
    Compute mean over axis 2.
    """
    return matrix.mean(axis=2)


def compute_global_mean(matrix: torch.Tensor) -> torch.Tensor:
    """
    Compute mean over both pairwise-distance axes.
    """
    return matrix.mean((1, 2))


def double_center_pairwise_distance_tensor(bpd: torch.Tensor) -> torch.Tensor:
    """
    Double-center a batch of pairwise-distance matrices.
    """
    row_mean = compute_row_mean(bpd)[:, None, :]
    col_mean = compute_column_mean(bpd)[:, :, None]
    global_mean = compute_global_mean(bpd)[:, None, None]
    return bpd - row_mean - col_mean + global_mean


def flatten_batch_pairwise_distance_tensor(bpd: torch.Tensor, n: int) -> torch.Tensor:
    """
    Flatten each centered pairwise-distance matrix to one row.
    """
    return bpd.view(n, -1)


def compute_inner_product_gram_from_centered_distances(
    bpd: torch.Tensor,
    n: int,
) -> torch.Tensor:
    """
    Compute the Gram matrix from flattened centered pairwise-distance tensors.
    """
    flat = flatten_batch_pairwise_distance_tensor(bpd, n)
    return torch.mm(flat, flat.T)


def free_distance_correlation_intermediates(*tensors) -> None:
    """
    Free intermediate tensors and trigger garbage collection.
    """
    for tensor in tensors:
        del tensor
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def divide_by_squared_sample_count(pd: torch.Tensor, n: int) -> torch.Tensor:
    """
    Apply the original 1/n^2 normalization.
    """
    pd /= n ** 2
    return pd


def take_elementwise_square_root(pd: torch.Tensor) -> torch.Tensor:
    """
    Take elementwise square root.
    """
    return torch.sqrt(pd)


def compute_diagonal_outer_sqrt_product(pd: torch.Tensor) -> torch.Tensor:
    """
    Compute sqrt(diag(pd)_i * diag(pd)_j) for all i,j.
    """
    return torch.sqrt(torch.diagonal(pd)[None, :] * torch.diagonal(pd)[:, None])


def normalize_by_distance_correlation_denominator(
    pd: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Normalize by the product of diagonal terms used in distance correlation.
    """
    denom = compute_diagonal_outer_sqrt_product(pd) + eps
    pd /= denom
    return pd


def set_matrix_diagonal_to_one(pd: torch.Tensor) -> torch.Tensor:
    """
    Set diagonal entries to 1 in place.
    """
    pd.fill_diagonal_(1)
    return pd


# can stay
def mat_discorr_adjacency(X: torch.tensor, Y: torch.tensor = None) -> torch.tensor:
    """
    Distance-correlation matrix calculation in tensor format. Return pairwise distance correlation among all row vectors in X.
    """
    n, m = X.shape
    Y = resolve_default_second_matrix(X, Y)
    bpd = compute_pairwise_feature_distances(X, Y)
    bpd = double_center_pairwise_distance_tensor(bpd)
    pd = compute_inner_product_gram_from_centered_distances(bpd, n)
    free_distance_correlation_intermediates(bpd, X, Y)

    pd = divide_by_squared_sample_count(pd, n)
    pd = take_elementwise_square_root(pd)
    pd = normalize_by_distance_correlation_denominator(pd, eps=1e-8)
    pd = set_matrix_diagonal_to_one(pd)

    return pd


# ---------------------------------------------------------------------
# Bhattacharyya-correlation helper math
# ---------------------------------------------------------------------

def validate_nonnegative_entries(X: torch.Tensor, message: str) -> None:
    """
    Validate that all tensor entries are nonnegative.
    """
    if torch.any(X < 0):
        raise ValueError(message)


def move_tensor_to_cuda(X: torch.Tensor) -> torch.Tensor:
    """
    Move a tensor to CUDA, matching the original behavior.
    """
    return X.cuda()


def compute_elementwise_square_root(X: torch.Tensor) -> torch.Tensor:
    """
    Compute elementwise square root of a tensor.
    """
    return torch.sqrt(X)


def compute_rowwise_similarity_by_matrix_product(X: torch.Tensor) -> torch.Tensor:
    """
    Compute row-by-row similarity via matrix multiplication with its transpose.
    """
    return torch.matmul(X, X.T)


# can stay
def mat_bc_adjacency(X):
    """
    Bhattacharyya correlation matrix version. Return pairwise BC correlation among all row vectors in X.
    """
    validate_nonnegative_entries(X, "Each value shoule in the range [0,1]")
    X = move_tensor_to_cuda(X)
    X_sqrt = compute_elementwise_square_root(X)
    return compute_rowwise_similarity_by_matrix_product(X_sqrt)


# ---------------------------------------------------------------------
# Cosine-similarity helper math
# ---------------------------------------------------------------------

def compute_rowwise_l2_norm(X: torch.Tensor) -> torch.Tensor:
    """
    Compute the L2 norm of each row as a column vector.
    """
    return torch.norm(X, p=2, dim=1).view(-1, 1)


def normalize_rows_by_l2_norm(
    X: torch.Tensor,
    norms: torch.Tensor,
    eps: float = 1e-4,
) -> torch.Tensor:
    """
    Normalize each row vector by its L2 norm.
    """
    return X / (norms + eps)


# can stay
def mat_cos_adjacency(X):
    """
    Cosine similarity matrix version. Return pairwise cos correlation among all row vectors in X.
    """
    X = move_tensor_to_cuda(X)
    X_row_l2_norm = compute_rowwise_l2_norm(X)
    X_row_std = normalize_rows_by_l2_norm(X, X_row_l2_norm, eps=1e-4)
    return compute_rowwise_similarity_by_matrix_product(X_row_std)


# ---------------------------------------------------------------------
# Pearson-correlation helper math
# ---------------------------------------------------------------------

def compute_rowwise_mean_as_column(X: torch.Tensor) -> torch.Tensor:
    """
    Compute rowwise mean and keep it as a column vector.
    """
    return X.mean(1).view(-1, 1)


def center_rows_by_rowwise_mean(X: torch.Tensor) -> torch.Tensor:
    """
    Center each row by subtracting its rowwise mean.
    """
    return X - compute_rowwise_mean_as_column(X)


def compute_covariance_like_matrix_from_centered_rows(X: torch.Tensor) -> torch.Tensor:
    """
    Compute row-wise covariance-like Gram matrix after centering.
    """
    return torch.matmul(X, X.T)


def create_cuda_scalar(value: float) -> torch.Tensor:
    """
    Create a CUDA scalar tensor.
    """
    return torch.tensor(value).cuda()


def compute_diagonal_square_root(cov: torch.Tensor) -> torch.Tensor:
    """
    Compute sqrt of the diagonal of a covariance-like matrix.
    """
    return torch.sqrt(torch.diagonal(cov))


def lower_bound_tensor_with_eps(
    values: torch.Tensor,
    eps_tensor: torch.Tensor,
) -> torch.Tensor:
    """
    Lower-bound values elementwise by eps.
    """
    return torch.maximum(values, eps_tensor)


def add_scalar_epsilon(values: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    """
    Add scalar epsilon to values.
    """
    return values + eps


def divide_covariance_by_row_and_column_scales(
    cov: torch.Tensor,
    sigma: torch.Tensor,
) -> torch.Tensor:
    """
    Convert covariance-like matrix to Pearson correlation matrix.
    """
    return cov / sigma.view(-1, 1) / sigma.view(1, -1)


# can stay
def mat_pearson_adjacency(X):
    """
    Pearson correlation matrix version. Return pairwise Pearson correlation among all row vectors in X.
    """
    X = move_tensor_to_cuda(X)
    X = center_rows_by_rowwise_mean(X)
    cov = compute_covariance_like_matrix_from_centered_rows(X)
    eps = create_cuda_scalar(1e-4)
    sigma = compute_diagonal_square_root(cov)
    sigma = lower_bound_tensor_with_eps(sigma, eps)
    sigma = add_scalar_epsilon(sigma, eps=1e-4)
    corr = divide_covariance_by_row_and_column_scales(cov, sigma)
    corr = set_matrix_diagonal_to_one(corr)
    return corr


# ---------------------------------------------------------------------
# Jensen-Shannon-divergence helper math
# ---------------------------------------------------------------------

def compute_pairwise_sum_tensor(X: torch.Tensor) -> torch.Tensor:
    """
    Compute the pairwise sum tensor paq used in the original JS-divergence implementation.
    """
    return X[:, :, None] + X.T[None, :, :]


def compute_log_with_epsilon(X: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    """
    Compute elementwise log(X + eps).
    """
    return torch.log(X + eps)


def halve_tensor(X: torch.Tensor) -> torch.Tensor:
    """
    Divide tensor by 2.
    """
    return X / 2


def compute_weighted_log_term(X: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    """
    Compute X * log(X + eps).
    """
    return X * torch.log(X + eps)


def sum_over_feature_axis(X: torch.Tensor) -> torch.Tensor:
    """
    Sum along axis 1, matching the original JS-divergence implementation.
    """
    return X.sum(1)


def diagonalize_vector(X: torch.Tensor) -> torch.Tensor:
    """
    Convert a vector to a diagonal matrix.
    """
    return torch.diag(X)


def flatten_tensor(X: torch.Tensor) -> torch.Tensor:
    """
    Flatten tensor to one dimension.
    """
    return X.flatten()


def compute_js_half_entropy_diagonal(
    paq: torch.Tensor,
    eps: float = 1e-4,
) -> torch.Tensor:
    """
    Compute the diagonal entropy-like term used in the original JS-divergence implementation.
    """
    half_paq = halve_tensor(paq)
    weighted_log = compute_weighted_log_term(half_paq, eps=eps)
    return flatten_tensor(diagonalize_vector(sum_over_feature_axis(weighted_log)))


def compute_js_cross_entropy_term(
    paq: torch.Tensor,
    eps: float = 1e-4,
) -> torch.Tensor:
    """
    Compute the cross term used in the original JS-divergence implementation.
    """
    half_paq = halve_tensor(paq)
    return sum_over_feature_axis(paq * compute_log_with_epsilon(half_paq, eps=eps))


def combine_js_terms_to_distance_matrix(
    paqdiag: torch.Tensor,
    cross_term: torch.Tensor,
) -> torch.Tensor:
    """
    Combine diagonal and cross terms into the final pairwise JS-divergence matrix.
    """
    return 1 / 2 * (paqdiag[:, None] + paqdiag[None, :] - cross_term)


# can stay
def mat_jsdiv_adjacency(X):
    """
    Jensen-Shannon Divergence matrix version. Return pairwise JS divergence among all row vectors in X.
    """
    validate_nonnegative_entries(X, "Each value should be in the range [0,1]")
    paq = compute_pairwise_sum_tensor(X)
    paqdiag = compute_js_half_entropy_diagonal(paq, eps=1e-4)
    cross_term = compute_js_cross_entropy_term(paq, eps=1e-4)
    return combine_js_terms_to_distance_matrix(paqdiag, cross_term)




def extract_original_layer_neuron_counts(module_list):
    conv_nfilters_list = extract_conv_input_counts(module_list)
    linear_nneurons_list = extract_linear_input_counts(module_list)
    return combine_layer_neuron_counts(conv_nfilters_list, linear_nneurons_list)

def topo_psf_feature_extract(
    model: torch.nn.Module,
    img_c,
    psf_config: Dict,
) -> Dict:
    """
    Point-cloud version of topo_psf_feature_extract.

    Expected model input shape:
        (B, 3, N)

    This function:
    1) creates or accepts a base point cloud,
    2) generates point-block perturbations,
    3) collects intermediate activations,
    4) builds a neuron correlation matrix,
    5) pools it to layer level,
    6) extracts 12 topological summary features (6 from H0, 6 from H1).

    Required external imports in this file:
        from ripser import Rips
        from pointcloud_helper import create_sample_pointcloud_for_tensor, gen_tensor_from_pointcloud
    """
    step_size = int(psf_config["step_size"])
    stim_level = int(psf_config["stim_level"])
    patch_size = int(psf_config["patch_size"])
    input_shape = psf_config["input_shape"]
    input_range = psf_config["input_range"]
    n_neuron_sample = int(psf_config["n_neuron"])
    corr_method = psf_config["corr_method"]
    device = psf_config["device"]

    model = model.to(device)
    model.eval()

    # ------------------------------------------------------------------
    # Base point cloud
    # ------------------------------------------------------------------
    if img_c is None:
        # We only need one base point cloud for stimulation.
        # Use the last entry of input_shape as number of points.
        n_points = int(input_shape[-1])
        base_pc_np = create_sample_pointcloud_for_tensor(B=1, N=n_points)
        base_pc = gen_tensor_from_pointcloud(base_pc_np).to(device)   # (1, 3, N)
    else:
        if isinstance(img_c, np.ndarray):
            base_pc = torch.FloatTensor(img_c).to(device)
        elif isinstance(img_c, torch.Tensor):
            base_pc = img_c.to(device)
        else:
            raise TypeError("img_c must be None, np.ndarray, or torch.Tensor")

        if base_pc.ndim == 2:
            # (3, N) -> (1, 3, N)
            base_pc = base_pc.unsqueeze(0)

    if base_pc.ndim != 3 or base_pc.shape[1] != 3:
        raise ValueError(f"Expected point cloud of shape (B, 3, N), got {tuple(base_pc.shape)}")

    _, _, n_points = base_pc.shape

    # If the run script still uses [0,255], override to a normalized point-cloud range.
    if (
        isinstance(input_range, (list, tuple))
        and len(input_range) == 2
        and float(input_range[1]) > 2.0
    ):
        stim_seq = np.linspace(-1.0, 1.0, stim_level, dtype=np.float32)
    else:
        stim_seq = np.linspace(float(input_range[0]), float(input_range[1]), stim_level, dtype=np.float32)

    # Forward once to infer number of classes
    with torch.no_grad():
        out0 = model(base_pc)
    num_classes = int(out0.shape[1])

    # ------------------------------------------------------------------
    # Point-block stimulation positions
    # ------------------------------------------------------------------
    point_positions = list(range(0, max(1, n_points - patch_size + 1), step_size))
    if len(point_positions) == 0:
        point_positions = [0]

    topo_feature_pos = []
    psf_feature_pos = []

    rips = Rips(verbose=False)
    layer_list, _ = parse_arch(model)

    # ------------------------------------------------------------------
    # For each stimulated point block, build one topological feature vector
    # ------------------------------------------------------------------
    for pos in point_positions:
        # Build batch of stimulated variants: (stim_level, 3, N)
        prob_input = base_pc.repeat(len(stim_seq), 1, 1)

        for s_idx, stim_val in enumerate(stim_seq):
            end_pos = min(pos + patch_size, n_points)
            prob_input[s_idx, :, pos:end_pos] = float(stim_val)

        with torch.no_grad():
            feature_dict_c, output = feature_collect(model, prob_input)

        # Save logits/confidence for compatibility (not used later in your run script)
        psf_score = output.detach().cpu()
        psf_conf = torch.nn.functional.softmax(psf_score, dim=1)
        psf_feature_pos.append(torch.stack([psf_score, psf_conf], dim=0))

        # --------------------------------------------------------------
        # Convert hooked tensors into per-neuron activation vectors
        # --------------------------------------------------------------
        neural_act = []

        for k in feature_dict_c:
            layer_tensor = feature_dict_c[k]

            # Conv1d input hook: usually (L, C, N)
            if layer_tensor.ndim == 3:
                # reduce over point dimension, keep one value per channel per stimulation
                layer_act = layer_tensor.max(dim=2)[0].T  # (C, L)

            # Linear input hook: usually (L, F)
            elif layer_tensor.ndim == 2:
                layer_act = layer_tensor.T  # (F, L)

            else:
                raise ValueError(
                    f"Unsupported hooked tensor shape {tuple(layer_tensor.shape)} for layer {k}"
                )

            layer_act = (
                layer_act - layer_act.mean(dim=1, keepdim=True)
            ) / (layer_act.std(dim=1, keepdim=True) + 1e-30)

            neural_act.append(layer_act)

        neural_act = torch.cat(neural_act, dim=0)  # (total_neurons, stim_level)

        if neural_act.shape[0] > n_neuron_sample:
            neural_act, sample_n_neurons_list = sample_act(
                neural_act, layer_list, sample_size=n_neuron_sample
            )
        else:
            sample_n_neurons_list = None

        # --------------------------------------------------------------
        # Correlation matrix
        # --------------------------------------------------------------
        if corr_method == "distcorr":
            neuron_pd = mat_discorr_adjacency(neural_act)
        elif corr_method == "bc":
            neuron_pd = mat_bc_adjacency(neural_act)
        elif corr_method == "cos":
            neuron_pd = mat_cos_adjacency(neural_act)
        elif corr_method == "pearson":
            neuron_pd = mat_pearson_adjacency(neural_act)
        elif corr_method == "js":
            neuron_pd = mat_jsdiv_adjacency(neural_act)
        else:
            raise ValueError(f"Unsupported corr_method: {corr_method}")

        neuron_pd = neuron_pd.detach().cpu()
        layer_pd = process_pd(neuron_pd, layer_list, sample_n_neurons_list=sample_n_neurons_list)

        # --------------------------------------------------------------
        # Convert similarity/correlation-like matrix to distance matrix
        # --------------------------------------------------------------
        dist_mat = 1.0 - np.array(layer_pd, dtype=np.float32)
        dist_mat = np.maximum(dist_mat, 0.0)
        np.fill_diagonal(dist_mat, 0.0)

        # --------------------------------------------------------------
        # Persistent homology
        # --------------------------------------------------------------
        dgms = rips.fit_transform(dist_mat, distance_matrix=True)

        def summarize_dgm(dgm: np.ndarray) -> List[float]:
            if dgm is None or len(dgm) == 0:
                return [0.0] * 6

            dgm = np.asarray(dgm, dtype=np.float32)
            finite_mask = np.isfinite(dgm[:, 0]) & np.isfinite(dgm[:, 1])
            dgm = dgm[finite_mask]

            if len(dgm) == 0:
                return [0.0] * 6

            births = dgm[:, 0]
            deaths = dgm[:, 1]
            pers = deaths - births
            mids = 0.5 * (births + deaths)

            return [
                float(np.max(pers)),          # max persistence
                float(np.mean(pers)),         # mean persistence
                float(np.std(pers)),          # std persistence
                float(np.mean(births)),       # mean birth
                float(np.mean(deaths)),       # mean death
                float(np.mean(mids)),         # mean midlife
            ]

        topo_feat = summarize_dgm(dgms[0]) + summarize_dgm(dgms[1])
        topo_feature_pos.append(torch.tensor(topo_feat, dtype=torch.float32))

    # Aggregate over all stimulation positions
    topo_feature_pos = torch.stack(topo_feature_pos, dim=0).mean(dim=0)   # (12,)

    if len(psf_feature_pos) > 0:
        psf_feature_pos = torch.stack(psf_feature_pos, dim=0)  # (n_pos, 2, stim_level, n_classes)
    else:
        psf_feature_pos = torch.empty(0)

    fv = {
        "psf_feature_pos": psf_feature_pos,
        "topo_feature_pos": topo_feature_pos,
    }
    return fv