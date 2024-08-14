import os
import gc

from configparser import ConfigParser, ExtendedInterpolation
import numpy as np
from numpy.typing import NDArray

import ngmix
import anacal
import fitsio
from metadetect import do_metadetect

from ..simulator.base import SimulateBase
from ..simulator.loader import MakeDMExposure
from .utils import get_psf_array, get_gridpsf_obj


def rotate90(image):
    rotated_image = np.zeros_like(image)
    rotated_image[1:, 1:] = np.rot90(m=image[1:, 1:], k=-1)
    return rotated_image


def parse_metadetect_config(cparser):
    # Create a ConfigParser object with extended interpolation
    # cparser = ConfigParser(interpolation=ExtendedInterpolation())

    # Read the configuration file
    # cparser.read(file_path)

    # Parse the configuration into a dictionary
    config_dict = {
        "meas_type": cparser["metadetect"].get("meas_type"),
        "metacal": {
            "psf": cparser["metadetect"].get("metacal.psf"),
            "types": [
                t.strip()
                for t in cparser["metadetect"].get("metacal.types").split(",")
            ],
        },
        "psf": {
            "model": cparser["metadetect"].get("psf.model"),
            "lm_pars": {},  # Assuming lm_pars is an empty dictionary as in the original config
            "ntry": cparser["metadetect"].getint("psf.ntry"),
        },
        "weight": {
            "fwhm": cparser["metadetect"].getfloat("weight.fwhm"),
        },
        "detect": {
            "thresh": cparser["metadetect"].getfloat("detect.thresh"),
        },
        "meds":{
            "box_padding": cparser["metadetect"].getint("meds.box_padding"),
            "box_type": cparser["metadetect"].get("meds.box_type"),
            "max_box_size": cparser["metadetect"].getint("meds.max_box_size"),
            "min_box_size": cparser["metadetect"].getint("meds.min_box_size"),
            "rad_fac": cparser["metadetect"].getfloat("meds.rad_fac"),
            "rad_min": cparser["metadetect"].getfloat("meds.rad_min"),
        }
    }

    return config_dict


def make_ngmix_obs(gal_array, psf_array, noise_array, pixel_scale, noise_std=0.37):
    """Transforms to Ngmix data

    Parameters:
        gal_array (ndarray):    galaxy array
        psf_array (ndarray):    psf array
    """

    cen = (np.array(gal_array.shape) - 1.0) / 2.0
    psf_cen = (np.array(psf_array.shape) - 1.0) / 2.0
    jacobian = ngmix.DiagonalJacobian(
        row=cen[0],
        col=cen[1],
        scale=pixel_scale,
    )
    psf_jacobian = ngmix.DiagonalJacobian(
        row=psf_cen[0],
        col=psf_cen[1],
        scale=pixel_scale,
    )

    gal_noise = noise_std
    noise_im = np.random.normal(size=gal_array.shape, scale=gal_noise)
    psf_noise = 1e-10
    wt = np.ones_like(gal_array) / gal_noise**2.0
    psf_wt = np.ones_like(psf_array) / psf_noise**2.0

    psf_obs = ngmix.Observation(
        psf_array,
        weight=psf_wt,
        jacobian=psf_jacobian,
    )
    #TODO: need to double check noise and noise image here is used correctly
    obs = ngmix.Observation(
        gal_array + noise_array,
        noise=noise_im,
        weight=wt,
        jacobian=jacobian,
        psf=psf_obs,
    )
    return obs


class ProcessSimMetadetect(SimulateBase):
    def __init__(self, config_name):
        cparser = ConfigParser(interpolation=ExtendedInterpolation())
        cparser.read(config_name)
        super().__init__(cparser)
        self.config_name = config_name

        if not os.path.isdir(self.img_dir):
            raise FileNotFoundError("Cannot find image directory")
        assert self.cat_dir is not None
        if not os.path.isdir(self.cat_dir):
            os.makedirs(self.cat_dir, exist_ok=True)

        # setup metadetect
        # parse the config into a dictionary for metadetect
        self.metadetect_config = parse_metadetect_config(cparser)

        self.ngrid = 2 * 32
        self.psf_rcut = 26

    def prepare_data(self, file_name):

        dm_task = MakeDMExposure(self.config_name)
        seed = dm_task.get_seed_from_fname(file_name, "i") + 1
        exposure = dm_task.generate_exposure(file_name)
        pixel_scale = float(exposure.getWcs().getPixelScale().asArcseconds())
        variance = np.average(exposure.getMaskedImage().variance.array)
        mag_zero = (
            np.log10(exposure.getPhotoCalib().getInstFluxAtZeroMagnitude())
            / 0.4
        )
        
        psf_obj = get_gridpsf_obj(
            exposure,
            ngrid=self.ngrid,
            psf_rcut=self.psf_rcut,
            dg=250,
        )

        psf_array = get_psf_array(
            exposure,
            ngrid=self.ngrid,
            psf_rcut=self.psf_rcut,
            dg=250,
        ).astype(np.float64)

        gal_array = np.asarray(
            exposure.getMaskedImage().image.array,
            dtype=np.float64,
        )
        mask_array = np.asanyarray(
            exposure.getMaskedImage().mask.array,
            dtype=np.int16,
        )

        del exposure, dm_task

        ny = self.coadd_dim + 10
        nx = self.coadd_dim + 10
        noise_std = np.sqrt(variance)
        
        # if noise is high enough     
        if variance > 1e-8:
            if self.corr_fname is None:
                # if no correlation file is provided, generate noise
                noise_array = (
                    np.random.RandomState(seed)
                    .normal(
                        scale=noise_std,
                        size=(ny, nx),
                    )
                    .astype(np.float64)
                )
            else:
                noise_corr = fitsio.read(self.corr_fname)
                noise_corr = rotate90(noise_corr)
                noise_array = (
                    anacal.noise.simulate_noise(
                        seed=seed,
                        correlation=noise_corr,
                        nx=nx,
                        ny=ny,
                        scale=pixel_scale,
                    ).astype(np.float64)
                    * noise_std
                )
        else:
            noise_array = None

        # TODO: not sure if we need the covariance matrix for metadetect
        # so leaving them as None for now
        cov_matrix = None

        if self.input_star_dir is not None:
            field_id = int(file_name.split("image-")[-1].split("_")[0])
            tmp_fname = "brightstar-%05d.fits" % field_id
            tmp_fname = os.path.join(self.input_star_dir, tmp_fname)
            star_cat = fitsio.read(tmp_fname)[["x", "y", "r"]]
        else:
            star_cat = None

        gc.collect()
        return {
            "gal_array": gal_array,
            "psf_array": psf_array,
            "cov_matrix": cov_matrix,
            "pixel_scale": pixel_scale,
            "noise_array": noise_array,
            "psf_obj": psf_obj,
            "mask_array": mask_array,
            "star_cat": star_cat,
        }


    def process_image(
        self,
        gal_array: NDArray,
        psf_array: NDArray,
        # cov_matrix: anacal.fpfs.table.Covariance,
        pixel_scale: float,
        noise_variance: float,
        noise_array: NDArray | None,
        # psf_obj: anacal.fpfs.BasePsf | None,
        # mask_array: NDArray,
        # star_cat: NDArray,
    ):

        obs = make_ngmix_obs(gal_array, psf_array, noise_array, pixel_scale)
        
        obs.mfrac = np.zeros_like(obs.image)
        
        mbobs = ngmix.MultiBandObsList()
        obslist = ngmix.ObsList()
        obslist.append(obs)
        mbobs.append(obslist)

        #TODO: need to think about the seed structure
        mdet_res = do_metadetect(self.metadetect_config, mbobs, np.random.RandomState(10))
        
        