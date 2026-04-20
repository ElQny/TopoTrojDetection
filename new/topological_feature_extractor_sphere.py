#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import torch

from typing import Dict, Optional
from ripser import Rips

from topo_utils import feature_collect, parse_arch, sample_act
from pointcloud_helper import (
    create_sample_pointcloud,
    center_and_scale,
    perturb_points_in_cube,
    possible_sphere_centers,
    generate_radius_batch,
)
from topological_feature_extractor_common import (
    generate_activation_vector_matrix,
    build_neural_correlation_matrix,
    build_persist_homology_vr,
    compute_topological_features,
    topo_feature_from_corr_matrix
)

def read_pointcloud_psf_config(psf_config: Dict):
    filtration_method = psf_config['filtration_method']
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
        filtration_method,
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


def topo_psf_feature_extract_sphere(model: torch.nn.Module, example_pointcloud: Optional[np.array], psf_config: Dict) -> Dict:
    """
        Combines all above functions as well as helper functions:
        - builds the pointcloud (without any example pointclouds)
        - generates perturbed pointclouds by cube-wise-perturbation
        - uses the DNN previously generated to get activation vectors
        - generates distance matrix from vectors
        - vectors are turned into topological features
    """

    (
    filtration_method,
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
    number_of_points_trigger) = read_pointcloud_psf_config(psf_config) #touple from function above

    if example_pointcloud is None:
        example_pointcloud = create_sample_pointcloud(number_of_points)
    else:
        example_pointcloud = np.array(example_pointcloud)
    example_pointcloud = center_and_scale(example_pointcloud)

    model = model.to(device)
    model.eval() # necessary for certain layers https://stackoverflow.com/questions/60018578/what-does-model-eval-do-in-pytorch

    centers = possible_sphere_centers(center_step)
    test_in, radii = generate_radius_batch(
        example_pointcloud,
        centers[0],
        radius_min,
        radius_max,
        radius_step,
        number_of_points_trigger
    ) #test_in = (B, 3, N)
    test_in = test_in.to(device)
    with torch.no_grad(): #reduce memory consumption (no Tensor.backward() calls here) https://docs.pytorch.org/docs/stable/generated/torch.no_grad.html
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

    #center-and-radius perturbation strategy:
    PD_list = []
    rips = Rips(verbose=False)
    model = model.to(device)

    for c_idx in range(len(centers)):
        print(f'Center #{c_idx}: {centers[c_idx]}')

        tensor, radii = generate_radius_batch(
            example_pointcloud,
            centers[c_idx],
            radius_min,
            radius_max,
            radius_step,
            number_of_points_trigger
        )
        tensor = tensor.to(device) #needs to be on same device as model for feature_collect

        feature_dict_c, output = feature_collect(model, tensor) #returns hooked activations and model output
        if isinstance(output, tuple):
            output = output[0]

        psf_score = output.detach().cpu()
        psf_conf = torch.softmax(psf_score, dim=1)

        psf_feature_pos[0,0,c_idx, :, :] = psf_score
        psf_feature_pos[1,0,c_idx, :,:] = psf_conf

        layer_list, _ = parse_arch(model)
        sample_n_neurons_list = None

        neural_act, layer_ids, layer_names = generate_activation_vector_matrix(feature_dict_c)  # hook-features -> neural activation matrix

        if len(neural_act) > 1.5e3:
            neural_act, sample_n_neurons_list, sample_indices = sample_act(neural_act, layer_list, sample_size=n_neuron_sample)
            layer_ids = layer_ids[sample_indices]
            layer_names = layer_names[sample_indices]

        neural_pd = build_neural_correlation_matrix(neural_act, method)  # Build neural correlation matrix (depending on correlation method)
        topo_feature = topo_feature_from_corr_matrix(PD_list=PD_list,
                                                     method=method,
                                                     neural_pd=neural_pd,
                                                     model=model,
                                                     rips=rips,
                                                     filtration_method=filtration_method,
                                                     layer_ids = layer_ids,
                                                     layer_names = layer_names,
                                                     )
        topo_feature_pos[c_idx, :] = topo_feature #overwrite where the feature is in the tensor (so topo_feature_pos stays at same size even with 0es)

    fv = {}
    fv['topo_feature_pos'] = topo_feature_pos
    fv['correlation_matrix'] = np.vstack([x[None, :, :] for x in PD_list]).mean(0)
    fv['psf_feature_pos'] = psf_feature_pos
    return fv
