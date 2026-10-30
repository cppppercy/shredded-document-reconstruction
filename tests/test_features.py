import numpy as np
from PIL import Image

from shredded_reconstruction.features import compatibility_features, pairwise_feature_tensor


def test_feature_shape_and_finiteness():
    array = np.tile(np.arange(80, dtype=np.uint8), (50, 1))
    image = Image.fromarray(array, mode="L")
    left, right = image.crop((0, 0, 40, 50)), image.crop((40, 0, 80, 50))
    features = compatibility_features(left, right)
    assert features.shape == (9,)
    assert np.all(np.isfinite(features))
    tensor = pairwise_feature_tensor([left, right])
    assert tensor.shape == (2, 2, 9)
