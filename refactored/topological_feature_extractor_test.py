import copy
import math
from collections import defaultdict

import numpy as np
import pytest
import torch

import topological_feature_extractor as tfe


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

class DummyModel(torch.nn.Module):
    """
    Minimal model with configurable number of classes.
    """
    def __init__(self, num_classes=3):
        super().__init__()
        self.num_classes = num_classes
        self.seen_inputs = []

    def forward(self, x):
        self.seen_inputs.append(x.detach().cpu().clone())
        batch = x.shape[0]
        # deterministic logits
        base = torch.arange(self.num_classes, dtype=torch.float32)
        return base.unsqueeze(0).repeat(batch, 1)


class DummyRips:
    """
    Fake Ripser wrapper that records calls and returns configurable PH.
    """
    def __init__(self, diagrams=None):
        self.calls = []
        self.diagrams = diagrams or [
            np.array([[0.0, 1.0], [0.2, np.inf]]),   # H0
            np.array([[0.5, 1.5]])                   # H1
        ]

    def fit_transform(self, D, distance_matrix=True):
        self.calls.append({
            "D": D,
            "distance_matrix": distance_matrix,
            "type": type(D).__name__,
        })
        return copy.deepcopy(self.diagrams)


def make_feature_dict_from_tensor_list(tensors):
    """
    Creates a feature_dict_c-like object with a single layer key.
    """
    return {("layer0", "Dummy"): tensors}


def simple_conv_like_features(batch_size, n_channels=2, h=2, w=2):
    """
    Produces list of tensors shaped like conv activations per sample.
    Each sample tensor is [C, H, W] after feature_dict_c[k][i].
    """
    out = []
    for i in range(batch_size):
        x = torch.arange(n_channels * h * w, dtype=torch.float32).reshape(n_channels, h, w)
        out.append(x + i)
    return out


def simple_linear_like_features(batch_size, n_features=4):
    """
    Produces list of tensors shaped like linear activations per sample.
    Each sample tensor is [F] after feature_dict_c[k][i].
    """
    out = []
    for i in range(batch_size):
        x = torch.arange(n_features, dtype=torch.float32) + i
        out.append(x)
    return out


# ---------------------------------------------------------------------
# Unit tests for helper functions
# ---------------------------------------------------------------------

def test_makeSparseDM_keeps_values_below_or_equal_threshold():
    D = np.array([
        [0.0, 0.3, 0.8],
        [0.3, 0.0, 0.6],
        [0.8, 0.6, 0.0],
    ])
    out = tfe.makeSparseDM(D, threshold=0.6)

    assert out.shape == (3, 3)
    dense = out.toarray()

    expected = np.array([
        [0.0, 0.3, 0.0],
        [0.3, 0.0, 0.6],
        [0.0, 0.6, 0.0],
    ])
    np.testing.assert_allclose(dense, expected)


def test_getGreedyPerm_matches_original_furthest_point_sampling_behavior():
    D = np.array([
        [0.0, 2.0, 5.0, 1.0],
        [2.0, 0.0, 3.0, 4.0],
        [5.0, 3.0, 0.0, 6.0],
        [1.0, 4.0, 6.0, 0.0],
    ])

    lambdas = tfe.getGreedyPerm(D)

    # Manual trace of original algorithm:
    # start ds = row 0 = [0,2,5,1]
    # choose idx=2 -> lambda 5
    # ds=min([0,2,5,1], row2=[5,3,0,6]) => [0,2,0,1]
    # choose idx=1 -> lambda 2
    # ds=min([0,2,0,1], row1=[2,0,3,4]) => [0,0,0,1]
    # choose idx=3 -> lambda 1
    # perm=[0,2,1,3], lambdas=[0,5,2,1], return lambdas[perm]=[0,2,5,1]
    expected = np.array([0.0, 2.0, 5.0, 1.0])
    np.testing.assert_allclose(lambdas, expected)


def test_getApproxSparseDM_returns_sparse_matrix():
    D = np.array([
        [0.0, 1.0, 4.0],
        [1.0, 0.0, 2.0],
        [4.0, 2.0, 0.0],
    ])
    lambdas = np.array([0.0, 1.0, 2.0])

    out = tfe.getApproxSparseDM(lambdas, eps=0.1, D=D.copy())

    from scipy.sparse import csr_matrix
    assert isinstance(out, csr_matrix)
    assert out.shape == (3, 3)


def test_calc_topo_feature_dim0_drops_last_point():
    PH = [
        np.array([
            [0.0, 1.0],
            [0.2, 0.7],
            [0.0, 9.0],   # should be dropped in dim 0
        ]),
        np.array([
            [0.5, 1.5],
        ]),
    ]

    feat0 = tfe.calc_topo_feature(PH, 0)

    # Only first two points remain
    persistence = np.array([1.0, 0.5])
    midlife = np.array([0.5, 0.45])

    assert feat0["betti_0"] == 2
    assert feat0["avepersis_0"] == pytest.approx(persistence.mean())
    assert feat0["avemidlife_0"] == pytest.approx(midlife.mean())
    assert feat0["maxmidlife_0"] == pytest.approx(np.median(midlife))
    assert feat0["maxpersis_0"] == pytest.approx(persistence.max())
    assert feat0["toppersis_0"] == pytest.approx(persistence.mean())  # [-5:] = all entries


def test_calc_topo_feature_empty_diagram_returns_zeros():
    PH = [
        np.empty((0, 2)),
        np.empty((0, 2)),
    ]

    feat1 = tfe.calc_topo_feature(PH, 1)

    assert feat1 == {
        "betti_1": 0,
        "avepersis_1": 0,
        "avemidlife_1": 0,
        "maxmidlife_1": 0,
        "maxpersis_1": 0,
        "toppersis_1": 0,
    }


# ---------------------------------------------------------------------
# Tests for topo_psf_feature_extract
# ---------------------------------------------------------------------

def base_psf_config():
    return {
        "step_size": 1,
        "stim_level": 4,
        "patch_size": 1,
        "input_shape": [1, 2, 3],   # C,H,W ; intentionally non-square to expose width bug
        "input_range": [0, 3],
        "n_neuron": 10,
        "corr_method": "distcorr",
        "device": torch.device("cpu"),
    }


def test_topo_psf_feature_extract_uses_blank_input_when_example_dict_is_falsy(monkeypatch):
    model = DummyModel(num_classes=3)
    cfg = base_psf_config()

    dummy_rips = DummyRips(diagrams=[
        np.array([[0.0, 1.0], [0.0, np.inf]]),
        np.array([[0.5, 2.0]])
    ])

    monkeypatch.setattr(tfe, "Rips", lambda verbose=False: dummy_rips)
    monkeypatch.setattr(tfe, "feature_collect", lambda model, x: (
        make_feature_dict_from_tensor_list(simple_linear_like_features(x.shape[0], 4)),
        model(x),
    ))
    monkeypatch.setattr(tfe, "parse_arch", lambda model: ([], []))
    monkeypatch.setattr(tfe, "sample_act", lambda neural_act, layer_list, sample_size: (neural_act, None))
    monkeypatch.setattr(tfe, "mat_discorr_adjacency", lambda x: torch.eye(x.shape[0]))

    out = tfe.topo_psf_feature_extract(model, {}, cfg)

    # One default example only
    assert out["psf_feature_pos"].shape[1] == 1
    assert out["topo_feature_pos"].shape[0] == 1


def test_topo_psf_feature_extract_output_shapes(monkeypatch):
    model = DummyModel(num_classes=5)
    cfg = base_psf_config()

    example_dict = defaultdict(list)
    example_dict[0].append(torch.zeros(cfg["input_shape"]).unsqueeze(0))
    example_dict[1].append(torch.ones(cfg["input_shape"]).unsqueeze(0))

    dummy_rips = DummyRips(diagrams=[
        np.array([[0.0, 1.0], [0.1, np.inf]]),
        np.array([[0.5, 1.5], [0.7, 1.1]])
    ])

    monkeypatch.setattr(tfe, "Rips", lambda verbose=False: dummy_rips)
    monkeypatch.setattr(tfe, "feature_collect", lambda model, x: (
        make_feature_dict_from_tensor_list(simple_linear_like_features(x.shape[0], 4)),
        model(x),
    ))
    monkeypatch.setattr(tfe, "parse_arch", lambda model: ([], []))
    monkeypatch.setattr(tfe, "sample_act", lambda neural_act, layer_list, sample_size: (neural_act, None))
    monkeypatch.setattr(tfe, "mat_discorr_adjacency", lambda x: torch.eye(x.shape[0]))

    out = tfe.topo_psf_feature_extract(model, example_dict, cfg)

    feature_map_h = len(range(0, cfg["input_shape"][1] - cfg["patch_size"] + 1, cfg["step_size"]))
    feature_map_w = len(range(0, cfg["input_shape"][2] - cfg["patch_size"] + 1, cfg["step_size"]))

    assert out["psf_feature_pos"].shape == (2, 2, feature_map_h, feature_map_w, cfg["stim_level"], 5)
    assert out["topo_feature_pos"].shape == (2, feature_map_h * feature_map_w, 12)
    assert out["correlation_matrix"].shape[0] == out["correlation_matrix"].shape[1]


@pytest.mark.parametrize(
    "method,expected_called",
    [
        ("distcorr", "distcorr"),
        ("bc", "bc"),
        ("cos", "cos"),
        ("pearson", "pearson"),
        ("js", "js"),
    ],
)
def test_topo_psf_feature_extract_dispatches_correlation_method(monkeypatch, method, expected_called):
    model = DummyModel(num_classes=2)
    cfg = base_psf_config()
    cfg["corr_method"] = method

    example_dict = defaultdict(list)
    example_dict[0].append(torch.zeros(cfg["input_shape"]).unsqueeze(0))

    dummy_rips = DummyRips()
    calls = []

    monkeypatch.setattr(tfe, "Rips", lambda verbose=False: dummy_rips)
    monkeypatch.setattr(tfe, "feature_collect", lambda model, x: (
        make_feature_dict_from_tensor_list(simple_linear_like_features(x.shape[0], 4)),
        model(x),
    ))
    monkeypatch.setattr(tfe, "parse_arch", lambda model: ([], []))
    monkeypatch.setattr(tfe, "sample_act", lambda neural_act, layer_list, sample_size: (neural_act, None))

    monkeypatch.setattr(tfe, "mat_discorr_adjacency", lambda x: calls.append("distcorr") or torch.eye(x.shape[0]))
    monkeypatch.setattr(tfe, "mat_bc_adjacency", lambda x: calls.append("bc") or torch.eye(x.shape[0]))
    monkeypatch.setattr(tfe, "mat_cos_adjacency", lambda x: calls.append("cos") or torch.eye(x.shape[0]))
    monkeypatch.setattr(tfe, "mat_pearson_adjacency", lambda x: calls.append("pearson") or torch.eye(x.shape[0]))
    monkeypatch.setattr(tfe, "mat_jsdiv_adjacency", lambda x: calls.append("js") or torch.eye(x.shape[0]))

    tfe.topo_psf_feature_extract(model, example_dict, cfg)

    assert expected_called in calls


def test_topo_psf_feature_extract_raises_on_unknown_correlation_method(monkeypatch):
    model = DummyModel(num_classes=2)
    cfg = base_psf_config()
    cfg["corr_method"] = "unknown_metric"

    example_dict = defaultdict(list)
    example_dict[0].append(torch.zeros(cfg["input_shape"]).unsqueeze(0))

    dummy_rips = DummyRips()

    monkeypatch.setattr(tfe, "Rips", lambda verbose=False: dummy_rips)
    monkeypatch.setattr(tfe, "feature_collect", lambda model, x: (
        make_feature_dict_from_tensor_list(simple_linear_like_features(x.shape[0], 4)),
        model(x),
    ))
    monkeypatch.setattr(tfe, "parse_arch", lambda model: ([], []))
    monkeypatch.setattr(tfe, "sample_act", lambda neural_act, layer_list, sample_size: (neural_act, None))

    with pytest.raises(Exception, match="isn't implemented"):
        tfe.topo_psf_feature_extract(model, example_dict, cfg)


def test_topo_psf_feature_extract_replaces_inf_in_persistence_diagrams(monkeypatch):
    model = DummyModel(num_classes=2)
    cfg = base_psf_config()

    example_dict = defaultdict(list)
    example_dict[0].append(torch.zeros(cfg["input_shape"]).unsqueeze(0))

    inf_diagrams = [
        np.array([[0.0, np.inf], [0.2, 0.8]]),
        np.array([[0.1, np.inf]])
    ]
    dummy_rips = DummyRips(diagrams=inf_diagrams)

    monkeypatch.setattr(tfe, "Rips", lambda verbose=False: dummy_rips)
    monkeypatch.setattr(tfe, "feature_collect", lambda model, x: (
        make_feature_dict_from_tensor_list(simple_linear_like_features(x.shape[0], 4)),
        model(x),
    ))
    monkeypatch.setattr(tfe, "parse_arch", lambda model: ([], []))
    monkeypatch.setattr(tfe, "sample_act", lambda neural_act, layer_list, sample_size: (neural_act, None))
    monkeypatch.setattr(tfe, "mat_discorr_adjacency", lambda x: torch.eye(x.shape[0]))

    out = tfe.topo_psf_feature_extract(model, example_dict, cfg)

    assert torch.isfinite(out["topo_feature_pos"]).all()


def test_topo_psf_feature_extract_calls_sample_act_only_above_threshold(monkeypatch):
    model = DummyModel(num_classes=2)
    cfg = base_psf_config()

    example_dict = defaultdict(list)
    example_dict[0].append(torch.zeros(cfg["input_shape"]).unsqueeze(0))

    dummy_rips = DummyRips()
    sample_calls = []

    # Make neural_act large enough: 1600 rows
    def fake_feature_collect(model, x):
        features = [torch.ones(1600) * i for i in range(x.shape[0])]
        return make_feature_dict_from_tensor_list(features), model(x)

    monkeypatch.setattr(tfe, "Rips", lambda verbose=False: dummy_rips)
    monkeypatch.setattr(tfe, "feature_collect", fake_feature_collect)
    monkeypatch.setattr(tfe, "parse_arch", lambda model: (["dummy_layer"], ["dummy_name"]))
    monkeypatch.setattr(
        tfe,
        "sample_act",
        lambda neural_act, layer_list, sample_size: sample_calls.append((neural_act.shape, sample_size)) or (neural_act[:10], [10]),
    )
    monkeypatch.setattr(tfe, "mat_discorr_adjacency", lambda x: torch.eye(x.shape[0]))

    tfe.topo_psf_feature_extract(model, example_dict, cfg)

    assert len(sample_calls) > 0
    assert sample_calls[0][1] == cfg["n_neuron"]


def test_topo_psf_feature_extract_does_not_call_sample_act_below_threshold(monkeypatch):
    model = DummyModel(num_classes=2)
    cfg = base_psf_config()

    example_dict = defaultdict(list)
    example_dict[0].append(torch.zeros(cfg["input_shape"]).unsqueeze(0))

    dummy_rips = DummyRips()
    sample_calls = []

    monkeypatch.setattr(tfe, "Rips", lambda verbose=False: dummy_rips)
    monkeypatch.setattr(tfe, "feature_collect", lambda model, x: (
        make_feature_dict_from_tensor_list(simple_linear_like_features(x.shape[0], 4)),
        model(x),
    ))
    monkeypatch.setattr(tfe, "parse_arch", lambda model: ([], []))
    monkeypatch.setattr(
        tfe,
        "sample_act",
        lambda neural_act, layer_list, sample_size: sample_calls.append(1) or (neural_act, None),
    )
    monkeypatch.setattr(tfe, "mat_discorr_adjacency", lambda x: torch.eye(x.shape[0]))

    tfe.topo_psf_feature_extract(model, example_dict, cfg)

    assert sample_calls == []


def test_topo_psf_feature_extract_uses_softmax_for_confidence(monkeypatch):
    model = DummyModel(num_classes=3)
    cfg = base_psf_config()

    example_dict = defaultdict(list)
    example_dict[0].append(torch.zeros(cfg["input_shape"]).unsqueeze(0))

    dummy_rips = DummyRips()

    def fake_feature_collect(model, x):
        batch = x.shape[0]
        logits = torch.tensor([[0.0, 1.0, 2.0]], dtype=torch.float32).repeat(batch, 1)
        return make_feature_dict_from_tensor_list(simple_linear_like_features(batch, 4)), logits

    monkeypatch.setattr(tfe, "Rips", lambda verbose=False: dummy_rips)
    monkeypatch.setattr(tfe, "feature_collect", fake_feature_collect)
    monkeypatch.setattr(tfe, "parse_arch", lambda model: ([], []))
    monkeypatch.setattr(tfe, "sample_act", lambda neural_act, layer_list, sample_size: (neural_act, None))
    monkeypatch.setattr(tfe, "mat_discorr_adjacency", lambda x: torch.eye(x.shape[0]))

    out = tfe.topo_psf_feature_extract(model, example_dict, cfg)

    expected_conf = torch.softmax(torch.tensor([0.0, 1.0, 2.0]), dim=0)
    actual_conf = out["psf_feature_pos"][1, 0, 0, 0, 0]
    torch.testing.assert_close(actual_conf, expected_conf)


# ---------------------------------------------------------------------
# Tests that lock in original quirks / bugs
# ---------------------------------------------------------------------

def test_original_bug_model_get_name_comparison_uses_attribute_not_call(monkeypatch):
    """
    Original code checks:
        if model._get_name=='ModdedLeNet5Net':
    instead of calling model._get_name().

    So even if the model class name would be ModdedLeNet5Net, this condition
    is False unless _get_name itself was overwritten with a string.
    """
    class FakeLeNet(DummyModel):
        pass

    model = FakeLeNet(num_classes=2)
    cfg = base_psf_config()

    example_dict = defaultdict(list)
    example_dict[0].append(torch.zeros(cfg["input_shape"]).unsqueeze(0))

    dummy_rips = DummyRips()
    approx_calls = []

    monkeypatch.setattr(tfe, "Rips", lambda verbose=False: dummy_rips)
    monkeypatch.setattr(tfe, "feature_collect", lambda model, x: (
        make_feature_dict_from_tensor_list(simple_linear_like_features(x.shape[0], 4)),
        model(x),
    ))
    monkeypatch.setattr(tfe, "parse_arch", lambda model: ([], []))
    monkeypatch.setattr(tfe, "sample_act", lambda neural_act, layer_list, sample_size: (neural_act, None))
    monkeypatch.setattr(tfe, "mat_discorr_adjacency", lambda x: torch.eye(x.shape[0]))
    monkeypatch.setattr(tfe, "getGreedyPerm", lambda D: approx_calls.append("getGreedyPerm") or np.zeros(D.shape[0]))
    monkeypatch.setattr(tfe, "getApproxSparseDM", lambda lambdas, eps, D: approx_calls.append("getApproxSparseDM") or D)

    tfe.topo_psf_feature_extract(model, example_dict, cfg)

    # Because of the bug, else-branch is used
    assert "getGreedyPerm" in approx_calls
    assert "getApproxSparseDM" in approx_calls


def test_original_width_bug_uses_input_shape_1_for_both_spatial_axes(monkeypatch):
    """
    Original code writes:
        int(pos_h):min(int(pos_h+patch_size), input_shape[1])

    instead of input_shape[2] for width.
    This is observable when H != W.
    """
    model = DummyModel(num_classes=2)
    cfg = base_psf_config()
    cfg["input_shape"] = [1, 2, 4]   # H=2, W=4
    cfg["patch_size"] = 2
    cfg["stim_level"] = 2
    cfg["input_range"] = [7, 9]

    example = torch.zeros(cfg["input_shape"]).unsqueeze(0)
    example_dict = defaultdict(list)
    example_dict[0].append(example)

    dummy_rips = DummyRips()
    captured_inputs = []

    def fake_feature_collect(model, x):
        captured_inputs.append(x.detach().cpu().clone())
        return make_feature_dict_from_tensor_list(simple_linear_like_features(x.shape[0], 4)), model(x)

    monkeypatch.setattr(tfe, "Rips", lambda verbose=False: dummy_rips)
    monkeypatch.setattr(tfe, "feature_collect", fake_feature_collect)
    monkeypatch.setattr(tfe, "parse_arch", lambda model: ([], []))
    monkeypatch.setattr(tfe, "sample_act", lambda neural_act, layer_list, sample_size: (neural_act, None))
    monkeypatch.setattr(tfe, "mat_discorr_adjacency", lambda x: torch.eye(x.shape[0]))
    monkeypatch.setattr(tfe, "getGreedyPerm", lambda D: np.zeros(D.shape[0]))
    monkeypatch.setattr(tfe, "getApproxSparseDM", lambda lambdas, eps, D: D)

    tfe.topo_psf_feature_extract(model, example_dict, cfg)

    first_batch = captured_inputs[0]
    # Because width bound incorrectly uses input_shape[1] == 2,
    # the written patch cannot extend beyond width index 1.
    # So columns >= 2 stay zero in first perturbation batch.
    assert torch.all(first_batch[:, :, :, 2:] == 0)


def test_original_batching_drops_tail_when_len_not_divisible_by_8(monkeypatch):
    """
    Original code with batch_size=8 uses:
        for b in range(int(len(prob_input)/batch_size))
    so it drops the remainder.
    """
    model = DummyModel(num_classes=2)
    cfg = base_psf_config()
    cfg["stim_level"] = 32  # >=32 so batch_size becomes 8 ;

    example_dict = defaultdict(list)
    example_dict[0].append(torch.zeros(cfg["input_shape"]).unsqueeze(0))

    dummy_rips = DummyRips()
    batch_sizes_seen = []

    def fake_feature_collect(model, x):
        batch_sizes_seen.append(x.shape[0])
        logits = torch.zeros(x.shape[0], 2)
        return make_feature_dict_from_tensor_list(simple_linear_like_features(x.shape[0], 4)), logits

    monkeypatch.setattr(tfe, "Rips", lambda verbose=False: dummy_rips)
    monkeypatch.setattr(tfe, "feature_collect", fake_feature_collect)
    monkeypatch.setattr(tfe, "parse_arch", lambda model: ([], []))
    monkeypatch.setattr(tfe, "sample_act", lambda neural_act, layer_list, sample_size: (neural_act, None))
    monkeypatch.setattr(tfe, "mat_discorr_adjacency", lambda x: torch.eye(x.shape[0]))
    monkeypatch.setattr(tfe, "getGreedyPerm", lambda D: np.zeros(D.shape[0]))
    monkeypatch.setattr(tfe, "getApproxSparseDM", lambda lambdas, eps, D: D)

    out = tfe.topo_psf_feature_extract(model, example_dict, cfg)

    # Per patch, only four batches of size 8 are processed => 32 predictions
    assert batch_sizes_seen[0] == 8
    assert sum(batch_sizes_seen[:4]) == 32
    assert out["psf_feature_pos"].shape[4] == 32
    # Stored tensor slice only has 32 rows of data available from concatenation in practice,
    # so this test mainly locks in the call pattern rather than assignment failure.


def test_original_assumes_example_dict_contains_key_zero(monkeypatch):
    """
    Original code does:
        test_input = example_dict[0][0].to(device)
    so if key 0 does not exist, it fails.
    """
    model = DummyModel(num_classes=2)
    cfg = base_psf_config()

    example_dict = defaultdict(list)
    example_dict[5].append(torch.zeros(cfg["input_shape"]).unsqueeze(0))

    with pytest.raises((KeyError, IndexError)):
        tfe.topo_psf_feature_extract(model, example_dict, cfg)


def test_original_uses_dictionary_keys_as_tensor_indices(monkeypatch):
    """
    Original code stores results via tensor index [c, ...], where c is the dict key.
    Non-contiguous / large keys can fail.
    """
    model = DummyModel(num_classes=2)
    cfg = base_psf_config()

    example_dict = defaultdict(list)
    example_dict[10].append(torch.zeros(cfg["input_shape"]).unsqueeze(0))
    example_dict[0].append(torch.zeros(cfg["input_shape"]).unsqueeze(0))

    dummy_rips = DummyRips()

    monkeypatch.setattr(tfe, "Rips", lambda verbose=False: dummy_rips)
    monkeypatch.setattr(tfe, "feature_collect", lambda model, x: (
        make_feature_dict_from_tensor_list(simple_linear_like_features(x.shape[0], 4)),
        model(x),
    ))
    monkeypatch.setattr(tfe, "parse_arch", lambda model: ([], []))
    monkeypatch.setattr(tfe, "sample_act", lambda neural_act, layer_list, sample_size: (neural_act, None))
    monkeypatch.setattr(tfe, "mat_discorr_adjacency", lambda x: torch.eye(x.shape[0]))
    monkeypatch.setattr(tfe, "getGreedyPerm", lambda D: np.zeros(D.shape[0]))
    monkeypatch.setattr(tfe, "getApproxSparseDM", lambda lambdas, eps, D: D)

    with pytest.raises(IndexError):
        tfe.topo_psf_feature_extract(model, example_dict, cfg)


def test_topological_feature_vector_order_is_sorted_key_order(monkeypatch):
    model = DummyModel(num_classes=2)
    cfg = base_psf_config()

    example_dict = defaultdict(list)
    example_dict[0].append(torch.zeros(cfg["input_shape"]).unsqueeze(0))

    diagrams = [
        np.array([[0.0, 1.0], [0.2, np.inf]]),  # H0 -> after drop, one point [0,1]
        np.array([[0.5, 1.0]])                  # H1
    ]
    dummy_rips = DummyRips(diagrams=diagrams)

    monkeypatch.setattr(tfe, "Rips", lambda verbose=False: dummy_rips)
    monkeypatch.setattr(tfe, "feature_collect", lambda model, x: (
        make_feature_dict_from_tensor_list(simple_linear_like_features(x.shape[0], 4)),
        model(x),
    ))
    monkeypatch.setattr(tfe, "parse_arch", lambda model: ([], []))
    monkeypatch.setattr(tfe, "sample_act", lambda neural_act, layer_list, sample_size: (neural_act, None))
    monkeypatch.setattr(tfe, "mat_discorr_adjacency", lambda x: torch.eye(x.shape[0]))
    monkeypatch.setattr(tfe, "getGreedyPerm", lambda D: np.zeros(D.shape[0]))
    monkeypatch.setattr(tfe, "getApproxSparseDM", lambda lambdas, eps, D: D)

    out = tfe.topo_psf_feature_extract(model, example_dict, cfg)

    vec = out["topo_feature_pos"][0, 0].numpy()

    feat0 = tfe.calc_topo_feature(
        [np.array([[0.0, 1.0], [0.2, 1.0]]), np.array([[0.5, 1.0]])], 0
    )
    feat1 = tfe.calc_topo_feature(
        [np.array([[0.0, 1.0], [0.2, 1.0]]), np.array([[0.5, 1.0]])], 1
    )

    expected = np.array([feat0[k] for k in sorted(feat0)] + [feat1[k] for k in sorted(feat1)])
    np.testing.assert_allclose(vec, expected)