import copy
import time
import numpy as np
import torch

from collections import defaultdict
from typing import Dict
from ripser import Rips

from topo_utils import feature_collect, parse_arch, sample_act
from topological_feature_extractor_common import (
    generate_activation_vector_matrix,
    build_neural_correlation_matrix,
    build_persist_homology_vr,
    compute_topological_features,
    topo_feature_from_corr_matrix
)

def read_img_psf_config(psf_config):
    filtration_method = psf_config['filtration_method']
    n_neuron_sample = psf_config['n_neuron']
    step_size = psf_config['step_size']
    stim_level = psf_config['stim_level']
    patch_size = psf_config['patch_size']
    input_shape = psf_config['input_shape']
    input_valuerange = psf_config['input_range']
    method = psf_config['corr_method']
    device = psf_config['device']
    return filtration_method, n_neuron_sample, device, input_shape, input_valuerange, method, patch_size, step_size, stim_level

def topo_psf_feature_extract_img(model: torch.nn.Module, example_dict: Dict, psf_config: Dict)-> Dict:
    """
    Extract topological features from a given torch model.
    Input args:
        model (torch.nn.Module). Target model.
        example_dict (Dict). Optional. Dictionary contains clean input examples. If None then all blank images are used.
    Return:
        fv (Dict). Dictionary contains extracted features
    """
    (filtration_method,
    n_neuron_sample,
    device,
    input_shape,
    input_valuerange,
    method,
    patch_size,
    step_size,
    stim_level) = read_img_psf_config(psf_config)
    # reads out parameters from psf_config (dictionary)

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
    # 12 is the number of topological features (including dim1 and dim2 features)
    topo_feature_pos=torch.zeros(
        len(example_dict.keys()),
        len(range(0, int(feature_map_h*feature_map_w))),
        12
    )
    # pixel-wise peturbation strategy:
    PH_list=[]
    PD_list=[]
    rips = Rips(verbose=False)
    model=model.to(device)
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

                layer_list, _ =parse_arch(model)
                sample_n_neurons_list=None

                # Extract intermediate activating vectors
                neural_act, layer_ids, layer_names = generate_activation_vector_matrix(feature_dict_c)

                if len(neural_act)>1.5e3:
                    neural_act, sample_n_neurons_list, sample_indices=sample_act(neural_act, layer_list, sample_size=n_neuron_sample)
                    layer_ids = layer_ids[sample_indices]
                    layer_names = layer_names[sample_indices]

                neural_pd = build_neural_correlation_matrix(neural_act, method = method)
                topo_feature = topo_feature_from_corr_matrix(PD_list=PD_list,
                                                             method=method,
                                                             neural_pd=neural_pd,
                                                             model = model ,
                                                             rips = rips,
                                                             filtration_method = filtration_method,
                                                             layer_ids = layer_ids,
                                                             layer_names = layer_names)

                topo_feature_pos[c, int(feature_w_pos*feature_map_w+feature_h_pos), :]=topo_feature
                feature_h_pos+=1
            feature_w_pos+=1

    fv={}
    fv['psf_feature_pos']=psf_feature_pos
    fv['topo_feature_pos']=topo_feature_pos
    fv['correlation_matrix']=np.vstack([x[None, :, :] for x in PD_list]).mean(0)
    return fv