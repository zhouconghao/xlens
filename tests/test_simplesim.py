"""xlens.simulator.simplesim: the isolated COSMOS simulation that used to
be anacal.simulation."""
import os

import galsim
import numpy as np
import pytest
from xlens.simulator import simplesim

pytestmark = pytest.mark.skipif(
    not os.path.isfile(
        os.path.join(
            os.environ.get("CATSIM_DIR", "."), simplesim.COSMOS_CATALOG_NAME
        )
    ),
    reason="needs $CATSIM_DIR/src_cosmos.fits",
)

PSF = galsim.Moffat(beta=3.5, fwhm=0.6, trunc=2.4)
KW = dict(
    gal_type="mixed", sim_method="fft", psf_obj=PSF, ny=256, nx=256,
    scale=0.2, nrot_per_gal=4, mag_zero=30.0,
)


def test_isolated_sim_shape_and_determinism():
    a = simplesim.make_isolated_sim(gname="g1-1", seed=3, buff=10, **KW)
    b = simplesim.make_isolated_sim(gname="g1-1", seed=3, buff=10, **KW)
    assert len(a) == 1
    img = a[0]
    assert img.shape == (276, 276)
    np.testing.assert_array_equal(img, b[0])
    # the zero-padding buffer stays empty and the galaxies carry flux
    assert np.all(img[:10] == 0) and np.all(img[:, :10] == 0)
    assert img.sum() > 0
    c = simplesim.make_isolated_sim(gname="g1-1", seed=4, buff=10, **KW)
    assert not np.array_equal(img, c[0])


def test_shear_versions_and_catalog():
    out, cat = simplesim.make_isolated_sim(
        gname="g2-0", seed=5, return_catalog=True, **KW
    )
    # 4x4 stamps of 64 px, 4 rotations per galaxy -> 4 input galaxies
    assert len(cat) == 4
    zero = simplesim.make_isolated_sim(gname="g2-2", seed=5, **KW)[0]
    assert not np.array_equal(out[0], zero)
    with pytest.raises(AssertionError):
        simplesim.make_isolated_sim(gname="g1-3", seed=5, **KW)
    with pytest.raises(ValueError):
        simplesim.make_isolated_sim(gname="g1-0", seed=5, **{**KW, "nx": 250})


def test_default_catalog_path(tmp_path):
    assert os.path.basename(
        simplesim.default_cosmos_catalog()
    ) == simplesim.COSMOS_CATALOG_NAME
    with pytest.raises(FileNotFoundError):
        simplesim.default_cosmos_catalog(str(tmp_path))
