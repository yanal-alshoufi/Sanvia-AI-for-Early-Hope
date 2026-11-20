import pytest
import torch

from sanvia.models.backbone_factory import build_backbone, get_backbone_spec, infer_feature_dim, resolve_weights, weights_metadata
from sanvia.models.multitask_model import build_model

NAMES = ["ResNet50", "DenseNet121", "InceptionV3", "MobileNetV3-Large", "EfficientNet-B4", "EfficientNetV2-S"]
EXPECTED_DIM = dict(zip(NAMES, [2048, 1024, 2048, 960, 1792, 1280]))
EXPECTED_SIZE = dict(zip(NAMES, [224, 224, 299, 224, 380, 384]))


@pytest.mark.parametrize("name", NAMES)
def test_spec_matches_protocol_and_weights_resolve(base_cfg, name):
    spec = get_backbone_spec(base_cfg, name)
    assert spec.input_size == EXPECTED_SIZE[name] and spec.feature_dim == EXPECTED_DIM[name]
    meta = weights_metadata(spec)  # resolves torchvision weights enum without downloading
    assert meta["weights_identifier"].endswith(spec.weights_name) and meta["mean"] == [0.485, 0.456, 0.406]
    assert resolve_weights(spec).name == spec.weights_name


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("use_age", [False, True])
def test_forward_shapes_and_feature_dim(base_cfg, name, use_age):
    spec = get_backbone_spec(base_cfg, name)
    model = build_model(spec, base_cfg["training"], use_age, pretrained=False).eval()
    size = 128 if name == "InceptionV3" else 96
    assert infer_feature_dim(model.backbone, size) == spec.feature_dim
    images = torch.randn(2, 2, 3, size, size)
    out = model(images, torch.randn(2, 2) if use_age else None)
    assert out["birads"].shape == (2, 5) and out["density"].shape == (2, 4)
    assert out["birads"].dtype == torch.float32


def test_age_input_required_when_enabled(base_cfg):
    model = build_model(get_backbone_spec(base_cfg, "MobileNetV3-Large"), base_cfg["training"], True, pretrained=False).eval()
    with pytest.raises(ValueError):
        model(torch.randn(1, 2, 3, 64, 64))


def test_views_share_backbone_and_are_concatenated(base_cfg):
    model = build_model(get_backbone_spec(base_cfg, "MobileNetV3-Large"), base_cfg["training"], False, pretrained=False).eval()
    x = torch.randn(3, 2, 3, 64, 64)
    with torch.no_grad():
        cc, mlo = model.backbone(x[:, 0]), model.backbone(x[:, 1])  # the same backbone weights process both views
        expected = model.birads_head(model.fusion(torch.cat([cc, mlo], dim=1)))
        assert torch.allclose(model(x)["birads"], expected, atol=1e-5)
        expected_swapped = model.birads_head(model.fusion(torch.cat([mlo, cc], dim=1)))
        assert torch.allclose(model(x.flip(1))["birads"], expected_swapped, atol=1e-5)  # order is [CC, MLO]


def test_five_dim_input_validation(base_cfg):
    model = build_model(get_backbone_spec(base_cfg, "MobileNetV3-Large"), base_cfg["training"], False, pretrained=False)
    with pytest.raises(ValueError):
        model(torch.randn(2, 3, 64, 64))


def test_freezing_and_gradual_unfreeze_and_bn_policy(base_cfg):
    model = build_model(get_backbone_spec(base_cfg, "ResNet50"), base_cfg["training"], False, pretrained=False)
    model.set_backbone_trainable(False)
    model.train()
    assert not any(p.requires_grad for p in model.backbone.parameters())
    assert all(not m.training for m in model.backbone.modules() if isinstance(m, torch.nn.BatchNorm2d))
    assert all(p.requires_grad for p in model.head_parameters())
    n = model.unfreeze_top_fraction(0.5)
    flags = [p.requires_grad for p in model.backbone.parameters()]
    assert n == sum(flags) and flags[-1] and not flags[0]
    model.unfreeze_top_fraction(1.0)
    assert all(p.requires_grad for p in model.backbone.parameters())


def test_inception_without_aux_returns_tensor(base_cfg):
    bb = build_backbone(get_backbone_spec(base_cfg, "InceptionV3"), pretrained=False).train()
    assert isinstance(bb(torch.randn(2, 3, 128, 128)), torch.Tensor)
