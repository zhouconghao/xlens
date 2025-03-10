# This file is part of pipe_tasks.
#
# Developed for the LSST Data Management System.
# This product includes software developed by the LSST Project
# (https://www.lsst.org).
# See the COPYRIGHT file at the top-level directory of this distribution
# for details of code ownership.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

__all__ = [
    "HaloMcBiasMultibandPipeConfig",
    "HaloMcBiasMultibandPipe",
    "HaloMcBiasMultibandPipeConnections",
]

from typing import Any

import lsst.pipe.base.connectionTypes as cT
import matplotlib.pyplot as plt
import numpy as np
from lsst.pex.config import Field
from lsst.pipe.base import (
    PipelineTask,
    PipelineTaskConfig,
    PipelineTaskConnections,
    Struct,
)
from lsst.skymap import BaseSkyMap
from lsst.utils.logging import LsstLogAdapter
from scipy.spatial import cKDTree


class HaloMcBiasMultibandPipeConnections(
    PipelineTaskConnections,
    dimensions=("skymap", "band"),
    defaultTemplates={
        "coaddName": "deep",
        "dataType": "",
    },
):
    skymap = cT.Input(
        doc="SkyMap to use in processing",
        name=BaseSkyMap.SKYMAP_DATASET_TYPE_NAME,
        storageClass="SkyMap",
        dimensions=("skymap",),
    )
    src00List = cT.Input(
        doc="Source catalog with all the measurement generated in this task",
        name="{coaddName}_0_rot0_Coadd_anacal_{dataType}",
        dimensions=(
            "skymap",
            "tract",
            "patch",
        ),
        storageClass="ArrowAstropy",
        multiple=True,
        deferLoad=True,
    )

    src01List = cT.Input(
        doc="Source catalog with all the measurement generated in this task",
        name="{coaddName}_0_rot1_Coadd_anacal_{dataType}",
        dimensions=(
            "skymap",
            "tract",
            "patch",
        ),
        storageClass="ArrowAstropy",
        multiple=True,
        deferLoad=True,
    )

    truth00List = cT.Input(
        doc="input truth catalog",
        name="{coaddName}_0_rot0_Coadd_truthCatalog",
        storageClass="ArrowAstropy",
        dimensions=(
            "skymap",
            "band",
            "tract",
            "patch",
        ),
        multiple=True,
        deferLoad=True,
    )

    truth01List = cT.Input(
        doc="input truth catalog",
        name="{coaddName}_0_rot1_Coadd_truthCatalog",
        storageClass="ArrowAstropy",
        dimensions=(
            "skymap",
            "band",
            "tract",
            "patch",
        ),
        multiple=True,
        deferLoad=True,
    )

    outputSummary = cT.Output(
        doc="Summary statistics",
        name="{coaddName}_halo_mc_{dataType}_summary_stats",
        storageClass="ArrowAstropy",
        dimensions=("skymap",),
    )

    summaryPlot = cT.Output(
        doc="simple plot of summary stats",
        storageClass="Plot",
        name="halo_mc_summary_{dataType}_plot",
        dimensions=("skymap",),
    )

    def __init__(self, *, config=None):
        super().__init__(config=config)


class HaloMcBiasMultibandPipeConfig(
    PipelineTaskConfig,
    pipelineConnections=HaloMcBiasMultibandPipeConnections,
):
    ename = Field[str](
        doc="ellipticity column name",
        default="e",
    )

    xname = Field[str](
        doc="detection coordinate row name",
        default="x",
    )

    yname = Field[str](
        doc="detection coordinate column name",
        default="y",
    )
    mass = Field[float](
        doc="halo mass",
        default=5e-14,
    )

    conc = Field[float](
        doc="halo concertration",
        default=1.0,
    )

    z_lens = Field[float](
        doc="halo redshift",
        default=1.0,
    )

    z_source = Field[float](
        doc="source redshift",
        default=None,
    )

    def validate(self):
        super().validate()
        if len(self.connections.dataType) == 0:
            raise ValueError("connections.dataTape missing")


class HaloMcBiasMultibandPipe(PipelineTask):
    _DefaultName = "HaloMcTask"
    ConfigClass = HaloMcBiasMultibandPipeConfig

    def __init__(
        self,
        *,
        config: HaloMcBiasMultibandPipeConfig | None = None,
        log: LsstLogAdapter | None = None,
        initInputs: dict[str, Any] | None = None,
        **kwargs: Any,
    ):
        super().__init__(
            config=config,
            log=log,
            initInputs=initInputs,
            **kwargs,
        )
        assert isinstance(
            self.config,
            HaloMcBiasMultibandPipeConfig,
        )

        self.ename = self.config.ename
        self.egname = lambda x, y: (
            "fpfs_d" + self.ename + str(x) + "_" + "dg" + str(y)
        )
        self.xname = self.config.xname
        self.yname = self.config.yname
        return

    def runQuantum(self, butlerQC, inputRefs, outputRefs):
        assert isinstance(
            self.config,
            HaloMcBiasMultibandPipeConfig,
        )
        # Retrieve the filename of the input exposure
        inputs = butlerQC.get(inputRefs)
        outputs = self.run(**inputs)
        butlerQC.put(outputs, outputRefs)
        return

    @staticmethod
    def _rotate_spin_2_vec(e1, e2, angle):
        """
        Rotate a spin-2 field by an array of angles (one per e1, e2 pair)
        """
        # Ensure e1, e2, and angle are numpy arrays of the same length
        e1 = np.asarray(e1)
        e2 = np.asarray(e2)
        angle = np.asarray(angle)

        assert (
            e1.shape == e2.shape == angle.shape
        ), "e1, e2, and angle must have the same shape"

        # Create an empty output array for the rotated values
        output = np.zeros((2, len(e1)))

        # Compute cos(2*angle) and sin(2*angle) for each angle
        cos_2angle = np.cos(2 * angle)
        sin_2angle = np.sin(2 * angle)

        # Apply the rotation for each e1, e2 pair
        # invert the sign of output so the tangential shear is positive
        output[0] = (cos_2angle * e1 - sin_2angle * e2) * (-1)  # Rotated e1
        output[1] = (sin_2angle * e1 + cos_2angle * e2) * (-1)  # Rotated e2

        return output

    @staticmethod
    def _rotate_spin_2_matrix(R11, R22, angle):
        # off diagonal terms are assumed to be zero
        # R is active rotation matrix

        output = np.zeros((2, len(R11)))

        output[0] = np.cos(2 * angle) ** 2 * R11 + np.sin(2 * angle) ** 2 * R22
        output[1] = np.sin(2 * angle) ** 2 * R11 + np.cos(2 * angle) ** 2 * R22

        return output

    @staticmethod
    def _get_response_from_w_and_der(e1, e2, w, e1_g1, e2_g2, w_g1, w_g2):
        r11 = e1_g1 * w + e1 * w_g1
        r22 = e2_g2 * w + e2 * w_g2

        return r11, r22

    @staticmethod
    def _get_angle_from_pixel(x, y, x_cen, y_cen):
        """
        Get the angle from the pixel coordinates
        the output is in radians between -pi and pi
        """
        return np.arctan2(y - y_cen, x - x_cen)

    @staticmethod
    def _get_eT_eX_rT_rX_sum(
        eT,
        eX,
        w,
        rT,
        rX,
        gT_true,
        gX_true,
        kappa_true,
        dist,
        lensed_shift,
        radial_lensed_shift,
        radial_bin_edges,
        match_dist,
        m00,
        m20,
    ):
        """calculate the sum of eT, eX, and rT in each radial bin for a single
        halo

        Args:
            eT (array): tangential shape
            eX (array): cross shape
            w (weight): ancal weight
            e1_g1 (array): partial derivative of e1 with respect to g1
            e2_g2 (array): partial derivative of e2 with respect to g2
            rT (array): tangential response,
            rX (array): cross response,
            gT_true (array): true tangential shear
            gX_true (array): true cross shear
            kappa_true (array): true convergence
            dist (array): pixel distance from the halo center
            lensed_shift (array):
                distance between the lensed and prelensed position
            radial_lensed_shift (array):
                distance between the lensed and prelensed position in radial
                direction
            radial_bin_edges (array): radial bin edges in pixel
            match_dist (array):
                the distance between detection and matched input
            m00 (array): fpfs shapelet mode m00
            m20 (array): fpfs shapelet mode m20

        Returns:
            eT(array): sum of eT in each radial bin
            eX(array): sum of eX in each radial bin
            rT(array): sum of tangential resposne in each radial bin
            rX(array): sum of radial response in each radial bin
            gT_true(array): sum of true tangential shear in each radial bin
            gX_true(array): sum of true cross shear in each radial bin
            kappa_true(array): sum of true convergence in each radial bin
            lensed_shift(array): mean of lensed shift in each radial bin
            radial_lensed_shift(array):
                mean of lensed shift in each radial bin, projected on the
                radial direction
            r_weighted_gT(array):
                sum of tangential shear weighted by rT in each radial bin
            r_weighted_gX(array):
                sum of cross shear weighted by rX in each radial bin
            ngal_in_bin(array): number of galaxies in each radial bin
            eT_std_list(array):
                per galaxy standard deviation of eT in each radial bin
            eX_std_list(array):
                per galaxy standard deviation of eX in each radial bin
            median_match_dist_list(array):
                median of the match distance in each radial bin, expected to be
                around 0.5
            match_failure_rate_list(array):
                fraction of match distance larger than 2 in each radial bin
        """

        n_bins = len(radial_bin_edges) - 1
        # this list stores results in each radial bin
        eT_list = []
        eX_list = []
        rT_list = []
        rX_list = []
        gT_true_list = []
        gX_true_list = []
        kappa_true_list = []
        r_weighted_gT_list = []
        r_weighted_gX_list = []
        ngal_in_bin = []
        eT_std_list = []
        eX_std_list = []
        lensed_shift_list = []
        radial_lensed_shift_list = []
        median_matched_dist_list = []
        match_failure_rate_list = []
        m00_list = []
        m20_list = []

        for i_bin in range(n_bins):
            mask = (dist >= radial_bin_edges[i_bin]) & (
                dist < radial_bin_edges[i_bin + 1]
            )
            eT_sum = np.sum(eT[mask] * w[mask])
            eX_sum = np.sum(eX[mask] * w[mask])
            # we use the mean of R11 and R22 as an estimator of Rt

            rT_sum = np.sum(rT[mask])
            rX_sum = np.sum(rX[mask])

            gT_true_list.append(np.sum(gT_true[mask]))
            gX_true_list.append(np.sum(gX_true[mask]))
            kappa_true_list.append(np.sum(kappa_true[mask]))
            r_weighted_gT_list.append(np.sum(gT_true[mask] * rT[mask]))
            r_weighted_gX_list.append(np.sum(gX_true[mask] * rX[mask]))
            ngal_in_bin.append(np.sum(mask))

            eT_list.append(eT_sum)
            eX_list.append(eX_sum)
            rT_list.append(rT_sum)
            rX_list.append(rX_sum)

            eT_std_list.append(np.std(eT[mask]))
            eX_std_list.append(np.std(eX[mask]))

            lensed_shift_list.append(np.mean(lensed_shift[mask]))
            radial_lensed_shift_list.append(np.mean(radial_lensed_shift[mask]))

            median_matched_dist_list.append(np.median(match_dist[mask]))
            match_failure_rate_list.append(
                np.sum(match_dist[mask] > 2) / np.sum(mask)
            )

            m00_list.append(np.mean(m00[mask]))
            m20_list.append(np.mean(m20[mask]))

        return (
            np.array(eT_list),
            np.array(eX_list),
            np.array(rT_list),
            np.array(rX_list),
            np.array(gT_true_list),
            np.array(gX_true_list),
            np.array(kappa_true_list),
            np.array(lensed_shift_list),
            np.array(radial_lensed_shift_list),
            np.array(r_weighted_gT_list),
            np.array(r_weighted_gX_list),
            np.array(ngal_in_bin),
            np.array(eT_std_list),
            np.array(eX_std_list),
            np.array(median_matched_dist_list),
            np.array(match_failure_rate_list),
            np.array(m00_list),
            np.array(m20_list),
        )

    @staticmethod
    def _match_input_to_det(true_x, true_y, det_x, det_y):
        input_coord = np.array([true_x, true_y]).T
        det_coord = np.array([det_x, det_y]).T
        # Create a cKDTree for input_coord
        tree = cKDTree(input_coord)
        # Query the nearest neighbors in the tree for each point in det_coord
        distance, idx = tree.query(det_coord, distance_upper_bound=100)

        # Check if the median match are within the threshold
        assert (
            np.median(distance) <= 5
        ), f"distance is too large, max distance is {np.max(distance)}"

        return idx, distance

    @staticmethod
    def get_summary_struct(n_halos, n_bins):
        dt = [
            (
                "angular_bin_left",
                f"({n_bins},)f8",
            ),
            (
                "angular_bin_right",
                f"({n_bins},)f8",
            ),
            ("ngal_in_bin", f"({n_bins},)i4"),
            ("eT", f"({n_bins},)f8"),
            ("eT_std", f"({n_bins},)f8"),
            ("eX", f"({n_bins},)f8"),
            ("eX_std", f"({n_bins},)f8"),
            ("rT", f"({n_bins},)f8"),
            ("rT_std", f"({n_bins},)f8"),
            ("rX", f"({n_bins},)f8"),
            ("rX_std", f"({n_bins},)f8"),
            ("gT_true", f"({n_bins},)f8"),
            ("gX_true", f"({n_bins},)f8"),
            ("kappa_true", f"({n_bins},)f8"),
            ("lensed_shift", f"({n_bins},)f8"),
            (
                "radial_lensed_shift",
                f"({n_bins},)f8",
            ),
            ("r_weighted_gT", f"({n_bins},)f8"),
            ("r_weighted_gX", f"({n_bins},)f8"),
            (
                "median_match_dist",
                f"({n_bins},)f8",
            ),
            (
                "match_failure_rate",
                f"({n_bins},)f8",
            ),
            ("mean_m00", f"({n_bins},)f8"),
            ("mean_m20", f"({n_bins},)f8"),
        ]
        return np.zeros(n_halos, dtype=dt)

    @staticmethod
    def generate_summary_plot(summary_table):

        area = np.mean(
            np.pi
            * (
                (summary_table["angular_bin_right"] / 3600 * np.pi / 180.0) ** 2
                - (summary_table["angular_bin_left"] / 3600 * np.pi / 180.0)
                ** 2
            )
            * (60 * 180 / np.pi) ** 2,
            axis=0,
        )

        fig, axes = plt.subplots(6, 2, figsize=(24, 26))

        angular_bin_mid = (
            (
                np.mean(
                    summary_table["angular_bin_left"],
                    axis=0,
                )
                + np.mean(
                    summary_table["angular_bin_right"],
                    axis=0,
                )
            )
            / 2
            / 60
        )

        start = 0
        mean_eX = np.mean(summary_table["eX"], axis=0)
        mean_rX = np.mean(summary_table["rX"], axis=0)
        sigma_eX = np.std(summary_table["eX"], axis=0) / np.sqrt(
            len(summary_table)
        )
        sigma_rX = np.std(summary_table["rX"], axis=0) / np.sqrt(
            len(summary_table)
        )
        sigma_gX = np.sqrt(
            (sigma_eX / mean_rX) ** 2 + (mean_eX * sigma_rX / mean_rX**2) ** 2
        )

        true_gX = np.mean(summary_table["r_weighted_gX"], axis=0) / np.mean(
            summary_table["rX"], axis=0
        )

        axes[0, 0].errorbar(
            angular_bin_mid[start:],
            (mean_eX / mean_rX)[start:],
            label="measured gX",
            yerr=sigma_gX[start:],
        )
        axes[0, 0].plot(
            angular_bin_mid[start:],
            true_gX[start:],
            label="r weighted true gX",
        )
        axes[0, 0].set_ylim(-0.01, 0.01)
        axes[0, 0].set_ylabel(r"$g^X$")
        axes[0, 0].set_xlabel("arcmin")
        axes[0, 0].legend()

        axes[1, 0].errorbar(
            angular_bin_mid[start:],
            (mean_eX / mean_rX - true_gX)[start:],
            yerr=sigma_gX[start:],
        )
        axes[1, 0].set_ylabel(r"$g^X_m - g^X_t$")

        mean_eT = np.mean(summary_table["eT"], axis=0)
        mean_rT = np.mean(summary_table["rT"], axis=0)
        sigma_eT = np.std(summary_table["eT"], axis=0) / np.sqrt(
            len(summary_table)
        )
        sigma_rT = np.std(summary_table["rT"], axis=0) / np.sqrt(
            len(summary_table)
        )
        sigma_gT = np.sqrt(
            (sigma_eT / mean_rT) ** 2 + (mean_eT * sigma_rT / mean_rT**2) ** 2
        )
        true_gT = np.mean(summary_table["r_weighted_gT"], axis=0) / np.mean(
            summary_table["rT"], axis=0
        )

        axes[0, 1].errorbar(
            angular_bin_mid[start:],
            (mean_eT / mean_rT)[start:],
            label="measured gT",
            yerr=sigma_gT[start:],
        )

        axes[0, 1].plot(
            angular_bin_mid[start:],
            true_gT[start:],
            label="r weighted true gT",
        )
        axes[0, 1].legend()
        axes[0, 1].set_xlabel("arcmin")
        axes[0, 1].set_ylabel(r"$g^T$")

        axes[1, 1].errorbar(
            angular_bin_mid[start:],
            (mean_eT / mean_rT / true_gT - 1)[start:],
            label="true gT",
            yerr=(sigma_gT / np.abs(true_gT))[start:],
        )
        axes[1, 1].set_ylabel(r"$g^T_{m}/g^T_{t} - 1$")
        # axes[1,1].set_ylim(-0.15, 0.15)

        # axes[1,0].errorbar()

        axes[3, 0].errorbar(
            angular_bin_mid,
            np.mean(
                summary_table["kappa_true"],
                axis=0,
            )
            / np.mean(
                summary_table["ngal_in_bin"],
                axis=0,
            ),
            yerr=np.std(
                summary_table["kappa_true"],
                axis=0,
            )
            / np.sqrt(len(summary_table))
            / np.mean(
                summary_table["ngal_in_bin"],
                axis=0,
            ),
        )
        axes[3, 0].set_ylabel(r"$\langle \kappa \rangle$")
        axes[3, 0].set_xlabel("arcmin")

        axes[2, 0].errorbar(
            angular_bin_mid,
            np.mean(summary_table["rX"], axis=0)
            / np.mean(
                summary_table["ngal_in_bin"],
                axis=0,
            ),
            yerr=np.std(summary_table["rX"], axis=0)
            / np.sqrt(len(summary_table))
            / np.mean(
                summary_table["ngal_in_bin"],
                axis=0,
            ),
        )
        axes[2, 0].set_xlabel("arcmin")
        axes[2, 0].set_ylabel(r"$\langle r^X \rangle$")
        # axes[1, 0].set_ylim(0.16, 0.185)

        axes[2, 1].errorbar(
            angular_bin_mid,
            np.mean(summary_table["rT"], axis=0)
            / np.mean(
                summary_table["ngal_in_bin"],
                axis=0,
            ),
            yerr=np.std(summary_table["rT"], axis=0)
            / np.sqrt(len(summary_table))
            / np.mean(
                summary_table["ngal_in_bin"],
                axis=0,
            ),
        )
        axes[2, 1].set_xlabel("arcmin")
        axes[2, 1].set_ylabel(r"$\langle r^T \rangle$")
        # axes[1, 1].set_ylim(0.16, 0.185)

        axes[3, 1].errorbar(
            angular_bin_mid,
            np.mean(
                summary_table["ngal_in_bin"],
                axis=0,
            )
            / area,
            yerr=np.std(
                summary_table["ngal_in_bin"],
                axis=0,
            )
            / np.sqrt(len(summary_table))
            / area,
            label=f"n_halo={len(summary_table)}",
        )
        axes[3, 1].set_xlabel("arcmin")
        axes[3, 1].set_ylabel(r"Detection Number density [arcmin$^{-2}$]")
        axes[3, 1].legend()

        axes[4, 0].errorbar(
            angular_bin_mid,
            np.mean(
                summary_table["lensed_shift"],
                axis=0,
            ),
            yerr=np.std(
                summary_table["lensed_shift"],
                axis=0,
            )
            / np.sqrt(len(summary_table)),
            label="shift",
        )
        axes[4, 0].errorbar(
            angular_bin_mid,
            np.mean(
                summary_table["radial_lensed_shift"],
                axis=0,
            ),
            yerr=np.std(
                summary_table["radial_lensed_shift"],
                axis=0,
            )
            / np.sqrt(len(summary_table)),
            label="radial_shift",
        )
        axes[4, 0].set_ylabel("Lens shift [arcsec]")
        axes[4, 0].legend()

        axes[4, 1].plot(
            angular_bin_mid,
            np.mean(
                summary_table["median_match_dist"],
                axis=0,
            ),
            label="match failure rate",
        )
        axes[4, 1].set_ylabel("Median match distance pixel")
        axes[4, 1].set_ylim(0, 0.5)

        axes[5, 0].plot(
            angular_bin_mid,
            np.mean(
                summary_table["match_failure_rate"],
                axis=0,
            ),
            label="match failure rate",
        )
        axes[5, 0].set_ylabel("Match failure rate (> 2 pixel)")
        axes[5, 0].set_ylim(0, 0.6)

        axes[5, 1].errorbar(
            angular_bin_mid,
            np.mean(
                summary_table["mean_m00"] + summary_table["mean_m20"],
                axis=0,
            ),
            yerr=(
                np.std(
                    summary_table["mean_m00"],
                    axis=0,
                )
                + np.std(
                    summary_table["mean_m20"],
                    axis=0,
                )
            )
            / np.sqrt(len(summary_table)),
        )
        axes[5, 1].set_ylabel(r"$M_{00} + M_{20}$")
        axes[5, 1].legend()

        for ax in axes.flatten():
            ax.set_xlim(0, 10)
            ax.set_xlabel("Angular separation [arcmin]")

        plt.subplots_adjust(hspace=0.2, wspace=0.2)

        return fig

    def run(
        self,
        skymap,
        src00List,
        src01List,
        truth00List,
        truth01List,
    ):

        assert skymap.config.patchBorder == 0, "patch border must be zero"

        self.log.info("load truth list")
        self.log.info(f"len truth00List: {len(truth00List)}")
        self.log.info(f"len truth01List: {len(truth01List)}")

        pixel_scale = skymap.config.pixelScale  # arcsec per pixel
        image_dim = skymap.config.patchInnerDimensions[0]  # in pixels

        max_pixel = (image_dim - 64) / 2

        self.log.info("image dim", image_dim)
        self.log.info("pixel scale", pixel_scale)

        self.log.info("max pixel", max_pixel)
        self.log.info(
            "max pixel in arcsec",
            max_pixel * pixel_scale,
        )

        n_bins = 10
        pixel_bin_edges = np.linspace(0, max_pixel, n_bins + 1)
        angular_bin_edges = pixel_bin_edges * pixel_scale

        en = self.ename
        e1n = "fpfs_" + en + "1"
        e2n = "fpfs_" + en + "2"

        e1g1n = self.egname(1, 1)
        e2g2n = self.egname(2, 2)

        self.log.info(
            "The length of source list is",
            len(src00List),
            len(src01List),
        )
        n_realization = len(src00List)

        rT_ensemble = np.empty((len(src00List), n_bins))
        rX_ensemble = np.empty((len(src00List), n_bins))
        eT_ensemble = np.empty((len(src00List), n_bins))
        eX_ensemble = np.empty((len(src00List), n_bins))
        gT_true_ensemble = np.empty((len(src00List), n_bins))
        gX_true_ensemble = np.empty((len(src00List), n_bins))
        r_weighted_gT_ensemble = np.empty((len(src00List), n_bins))
        r_weighted_gX_ensemble = np.empty((len(src00List), n_bins))
        kappa_true_ensemble = np.empty((len(src00List), n_bins))
        ngal_in_bin_ensemble = np.empty((len(src00List), n_bins))
        eT_std_ensemble = np.empty((len(src00List), n_bins))
        eX_std_ensemble = np.empty((len(src00List), n_bins))
        lensed_shift_ensemble = np.empty((len(src00List), n_bins))
        radial_lensed_shift_ensemble = np.empty((len(src00List), n_bins))
        median_match_dist_ensemble = np.empty((len(src00List), n_bins))
        match_failure_rate_ensemble = np.empty((len(src00List), n_bins))
        m00_ensemble = np.empty((len(src00List), n_bins))
        m20_ensemble = np.empty((len(src00List), n_bins))

        for i, cats in enumerate(
            zip(
                src00List,
                src01List,
                truth00List,
                truth01List,
            )
        ):
            src00, src01, truth00, truth01 = (
                cats[0],
                cats[1],
                cats[2],
                cats[3],
            )
            sr_00_res = src00.get()
            sr_01_res = src01.get()
            truth_00_res = truth00.get()
            truth_01_res = truth01.get()

            self.log.debug(
                "truth 00 x residual is %.3f"
                % (np.mean(truth_00_res["image_x"] - (image_dim) / 2))
            )
            self.log.debug(
                "truth 00 y residual is %.3f"
                % (np.mean(truth_00_res["image_y"] - (image_dim) / 2))
            )
            self.log.debug(
                "truth 01 x residual is %.3f"
                % (np.mean(truth_01_res["image_x"] - (image_dim) / 2))
            )
            self.log.debug(
                "truth 01 y residual is %.3f"
                % (np.mean(truth_01_res["image_y"] - (image_dim) / 2))
            )

            idx_00, match_dist_00 = self._match_input_to_det(
                truth_00_res["image_x"],
                truth_00_res["image_y"],
                sr_00_res["x"],
                sr_00_res["y"],
            )

            idx_01, match_dist_01 = self._match_input_to_det(
                truth_01_res["image_x"],
                truth_01_res["image_y"],
                sr_01_res["x"],
                sr_01_res["y"],
            )

            match_dist = np.concatenate([match_dist_00, match_dist_01])

            gamma1_true = np.concatenate(
                [
                    truth_00_res["gamma1"][idx_00],
                    truth_01_res["gamma1"][idx_01],
                ]
            )
            gamma2_true = np.concatenate(
                [
                    truth_00_res["gamma2"][idx_00],
                    truth_01_res["gamma2"][idx_01],
                ]
            )
            kappa_true = np.concatenate(
                [
                    truth_00_res["kappa"][idx_00],
                    truth_01_res["kappa"][idx_01],
                ]
            )
            g1_true = gamma1_true / (1 - kappa_true)
            g2_true = gamma2_true / (1 - kappa_true)

            e1 = np.concatenate([sr_00_res[e1n], sr_01_res[e1n]])
            e2 = np.concatenate([sr_00_res[e2n], sr_01_res[e2n]])
            self.log.info(f"i: {i}, e1: {e1.shape}, e2: {e2.shape}")
            e1_g1 = np.concatenate(
                [
                    sr_00_res[e1g1n],
                    sr_01_res[e1g1n],
                ]
            )
            e2_g2 = np.concatenate(
                [
                    sr_00_res[e2g2n],
                    sr_01_res[e2g2n],
                ]
            )
            w = np.concatenate([sr_00_res["fpfs_w"], sr_01_res["fpfs_w"]])
            w_g1 = np.concatenate(
                [
                    sr_00_res["fpfs_dw_dg1"],
                    sr_01_res["fpfs_dw_dg1"],
                ]
            )
            w_g2 = np.concatenate(
                [
                    sr_00_res["fpfs_dw_dg2"],
                    sr_01_res["fpfs_dw_dg2"],
                ]
            )
            m00 = np.concatenate(
                [
                    sr_00_res["fpfs_m00"],
                    sr_01_res["fpfs_m00"],
                ]
            )
            m20 = np.concatenate(
                [
                    sr_00_res["fpfs_m20"],
                    sr_01_res["fpfs_m20"],
                ]
            )

            # use the prelensed location in binning and calculating angle
            x = np.concatenate(
                [
                    truth_00_res["prelensed_image_x"][idx_00],
                    truth_01_res["prelensed_image_x"][idx_01],
                ]
            )
            y = np.concatenate(
                [
                    truth_00_res["prelensed_image_y"][idx_00],
                    truth_01_res["prelensed_image_y"][idx_01],
                ]
            )

            lensed_x = np.concatenate(
                [
                    truth_00_res["image_x"][idx_00],
                    truth_01_res["image_x"][idx_01],
                ]
            )
            lensed_y = np.concatenate(
                [
                    truth_00_res["image_y"][idx_00],
                    truth_01_res["image_y"][idx_01],
                ]
            )
            self.log.debug(
                f"mean x offset: {np.mean(lensed_x - (image_dim) / 2)},",
                f"mean y offset: {np.mean(lensed_y - (image_dim) / 2)}",
            )
            self.log.debug(
                f"prelensed mean x offset: {np.mean(x - (image_dim) / 2)},",
                f"prelensed mean y offset: {np.mean(y - (image_dim) / 2)}",
            )

            lensed_shift = (
                np.sqrt((lensed_x - x) ** 2 + (lensed_y - y) ** 2) * pixel_scale
            )
            radial_dist_lensed = np.sqrt(
                (lensed_x - (image_dim) / 2) ** 2
                + (lensed_y - (image_dim) / 2) ** 2
            )
            radial_dist = np.sqrt(
                (x - (image_dim) / 2) ** 2 + (y - (image_dim) / 2) ** 2
            )
            radial_lensed_shift = (
                radial_dist_lensed - radial_dist
            ) * pixel_scale

            self.log.info(
                f"mean radial lensed shift: {np.mean(radial_lensed_shift)}"
            )

            angle = self._get_angle_from_pixel(
                lensed_x,
                lensed_y,
                (image_dim) / 2,
                (image_dim) / 2,
            )
            # negative since we are rotating axes
            eT, eX = self._rotate_spin_2_vec(e1, e2, -angle)
            gT_true, gX_true = self._rotate_spin_2_vec(
                g1_true,
                g2_true,
                -angle,
            )
            # w are scalar so no need to rotate
            dist = np.sqrt(
                (lensed_x - (image_dim) / 2) ** 2
                + (lensed_y - (image_dim) / 2) ** 2
            )

            r11, r22 = self._get_response_from_w_and_der(
                e1,
                e2,
                w,
                e1_g1,
                e2_g2,
                w_g1,
                w_g2,
            )
            rT, rX = self._rotate_spin_2_matrix(r11, r22, angle)

            (
                eT_list,
                eX_list,
                rT_list,
                rX_list,
                gT_true_list,
                gX_true_list,
                kappa_true_list,
                lensed_shift_list,
                radial_lensed_shift_list,
                r_weighted_gT_list,
                r_weighted_gX_list,
                ngal_in_bin,
                eT_std_list,
                eX_std_list,
                median_match_dist,
                match_failure_rate,
                m00_list,
                m20_list,
            ) = self._get_eT_eX_rT_rX_sum(
                eT,
                eX,
                w,
                rT,
                rX,
                gT_true,
                gX_true,
                kappa_true,
                dist,
                lensed_shift,
                radial_lensed_shift,
                pixel_bin_edges,
                match_dist,
                m00,
                m20,
            )
            rT_ensemble[i, :] = rT_list
            rX_ensemble[i, :] = rX_list
            eT_ensemble[i, :] = eT_list
            eX_ensemble[i, :] = eX_list
            gT_true_ensemble[i, :] = gT_true_list
            gX_true_ensemble[i, :] = gX_true_list
            r_weighted_gT_ensemble[i, :] = r_weighted_gT_list
            r_weighted_gX_ensemble[i, :] = r_weighted_gX_list
            kappa_true_ensemble[i, :] = kappa_true_list
            lensed_shift_ensemble[i, :] = lensed_shift_list
            radial_lensed_shift_ensemble[i, :] = radial_lensed_shift_list
            ngal_in_bin_ensemble[i, :] = ngal_in_bin
            eT_std_ensemble[i, :] = eT_std_list / np.sqrt(ngal_in_bin)
            eX_std_ensemble[i, :] = eX_std_list / np.sqrt(ngal_in_bin)
            median_match_dist_ensemble[i, :] = median_match_dist
            match_failure_rate_ensemble[i, :] = match_failure_rate
            m00_ensemble[i, :] = m00_list
            m20_ensemble[i, :] = m20_list

        summary_stats = self.get_summary_struct(
            n_realization,
            len(angular_bin_edges) - 1,
        )

        # Populate the structured array directly with the ensemble variables
        summary_stats["angular_bin_left"] = np.tile(
            angular_bin_edges[:-1],
            (n_realization, 1),
        )
        summary_stats["angular_bin_right"] = np.tile(
            angular_bin_edges[1:],
            (n_realization, 1),
        )
        summary_stats["ngal_in_bin"] = (
            ngal_in_bin_ensemble  # Shape (n_halos, n_bins)
        )
        summary_stats["eT"] = eT_ensemble  # Shape (n_halos, n_bins)
        summary_stats["eT_std"] = eT_std_ensemble  # Shape (n_halos, n_bins)
        summary_stats["eX"] = eX_ensemble  # Shape (n_halos, n_bins)
        summary_stats["eX_std"] = eX_std_ensemble  # Shape (n_halos, n_bins)
        summary_stats["rT"] = rT_ensemble  # Shape (n_halos, n_bins)
        summary_stats["rX"] = rX_ensemble  # Shape (n_halos, n_bins)
        summary_stats["gT_true"] = gT_true_ensemble  # Shape (n_halos, n_bins)
        summary_stats["gX_true"] = gX_true_ensemble  # Shape (n_halos, n_bins)
        summary_stats["kappa_true"] = (
            kappa_true_ensemble  # Shape (n_halos, n_bins)
        )
        summary_stats["lensed_shift"] = (
            lensed_shift_ensemble  # Shape (n_halos, n_bins)
        )
        summary_stats["radial_lensed_shift"] = (
            radial_lensed_shift_ensemble  # Shape (n_halos, n_bins)
        )
        summary_stats["r_weighted_gT"] = (
            r_weighted_gT_ensemble  # Shape (n_halos, n_bins)
        )
        summary_stats["r_weighted_gX"] = (
            r_weighted_gX_ensemble  # Shape (n_halos, n_bins)
        )
        summary_stats["radial_lensed_shift"] = (
            radial_lensed_shift_ensemble  # Shape (n_halos, n_bins)
        )
        summary_stats["median_match_dist"] = (
            median_match_dist_ensemble  # Shape (n_halos, n_bins)
        )
        summary_stats["match_failure_rate"] = (
            match_failure_rate_ensemble  # Shape (n_halos, n_bins)
        )

        summary_stats["mean_m00"] = m00_ensemble
        summary_stats["mean_m20"] = m20_ensemble

        summary_plot = self.generate_summary_plot(summary_stats)

        return Struct(
            outputSummary=summary_stats,
            summaryPlot=summary_plot,
        )
