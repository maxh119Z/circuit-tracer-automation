import torch
import torch.nn as nn
from transformer_lens import HookedTransformerConfig

from circuit_tracer import attribute
from circuit_tracer.replacement_model import ReplacementModel
from circuit_tracer.transcoder.cross_layer_transcoder import CrossLayerTranscoder
from circuit_tracer.utils import get_default_device


def create_clt_model(cfg: HookedTransformerConfig):
    """Create a CLT and ReplacementModel with random weights."""
    # Create CLT with 4x expansion
    clt = CrossLayerTranscoder(
        n_layers=cfg.n_layers,
        d_transcoder=cfg.d_model * 4,
        d_model=cfg.d_model,
        dtype=cfg.dtype,
        lazy_decoder=False,
    )

    # Initialize CLT weights
    with torch.no_grad():
        for param in clt.parameters():
            nn.init.uniform_(param, a=-0.1, b=0.1)

    # Create model
    model = ReplacementModel.from_config(cfg, clt)

    # Monkey patch all_special_ids if necessary
    type(model.tokenizer).all_special_ids = property(lambda self: [0])  # type: ignore

    # Initialize model weights
    with torch.no_grad():
        for param in model.parameters():
            if param.requires_grad:
                nn.init.uniform_(param, a=-0.1, b=0.1)

    return model


def verify_feature_intervention(model, graph, feature_idx):
    """Verify that intervening on a feature produces the expected effects."""
    prompt = graph.input_tokens.unsqueeze(0)
    layer, pos, feat_id = graph.active_features[feature_idx]
    activation = graph.activation_values[feature_idx]
    influences = graph.adjacency_matrix[:, feature_idx]

    # Get decoder vectors for cross-layer effects
    decoder_vectors = model.transcoders.W_dec[layer][feat_id]

    def apply_steering(activations, hook):
        steer_layer = hook.layer() - layer
        activations[0, pos] += decoder_vectors[steer_layer] * activation
        return activations

    # Setup hooks
    cache, caching_hooks, _ = model.get_caching_hooks(
        names_filter=lambda name: model.feature_input_hook in name
    )
    freeze_hooks = model.setup_intervention_with_freeze(prompt, direct_effects=True)
    steering_hooks = [
        (f"blocks.{lyr}.{model.feature_output_hook}", apply_steering)
        for lyr in range(layer, model.cfg.n_layers)
    ]

    # Run intervention
    with model.hooks(freeze_hooks + caching_hooks + steering_hooks):
        _ = model(prompt)

    # Compute new activations
    clt_inputs = torch.cat(list(cache.values()), dim=0)
    new_activations = (
        torch.einsum("lbd,lfd->lbf", clt_inputs, model.transcoders.W_enc)
        + model.transcoders.b_enc[:, None]
    )
    new_activations = new_activations[tuple(graph.active_features.T)]

    # Calculate error
    delta = new_activations - graph.activation_values
    n_active = len(graph.active_features)
    expected_delta = influences[:n_active].to(get_default_device())

    max_error = (delta - expected_delta).abs().max().item()
    return max_error


def test_clt_attribution():
    """Test CLT attribution and intervention mechanism."""
    # Minimal config
    cfg = HookedTransformerConfig.from_dict(
        {
            "n_layers": 4,
            "d_model": 8,
            "n_ctx": 32,
            "d_head": 4,
            "n_heads": 2,
            "d_mlp": 32,
            "act_fn": "gelu",
            "d_vocab": 50,
            "model_name": "test-clt",
            "device": get_default_device(),
            "tokenizer_name": "gpt2",
        }
    )

    # Create model
    model = create_clt_model(cfg)

    # Run attribution
    prompt = torch.tensor([[0, 1, 2, 3, 4, 5]])
    graph = attribute(prompt, model, max_n_logits=5, desired_logit_prob=0.8, batch_size=32)

    # Test interventions on multiple random features
    n_active = len(graph.active_features)
    n_samples = min(100, n_active)
    sample_indices = torch.randperm(n_active)[:n_samples]

    max_errors = []

    from tqdm import tqdm

    for idx in tqdm(sample_indices):
        max_error = verify_feature_intervention(model, graph, idx)
        max_errors.append(max_error)

        assert max_error < 5e-4, f"Feature {idx}: max error {max_error:.6f} exceeds threshold"

    mean_error = sum(max_errors) / len(max_errors)
    max_error = max(max_errors)

    print("✓ CLT attribution test passed!")
    print(f"  Tested {n_samples} features out of {n_active} active features")
    print(f"  Mean max error: {mean_error:.6f}, Worst error: {max_error:.6f}")
