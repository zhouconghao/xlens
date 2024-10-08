#!/usr/bin/env python
#
# FPFS shear estimator
# Copyright 20220312 Xiangchong Li.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
import glob
import os
import time
from configparser import ConfigParser, ExtendedInterpolation

import fitsio
import numpy as np
from anacal.fpfs import CatalogTask
from anacal.fpfs.table import Catalog, Covariance

from ..simulator.base import SimulateBatchBase

pf = {
    "snr_min": 1.0,
    "r2_min": 100.0,
    "r2_max": 100.0,
}


class SummarySimFpfs(SimulateBatchBase):
    def __init__(
        self,
        config_name,
        min_id=0,
        max_id=1000,
        ncores=1,
    ):
        cparser = ConfigParser(interpolation=ExtendedInterpolation())
        cparser.read(config_name)
        super().__init__(cparser, min_id, max_id, ncores)
        assert self.cat_dir is not None
        if not os.path.isdir(self.cat_dir):
            raise FileNotFoundError(
                "Cannot find catalog directory: %s" % self.cat_dir
            )

        self.shear_comp_sim = cparser.get(
            "simulation",
            "shear_component",
            fallback="g1",
        )
        # FPFS parameters
        self.norder = cparser.getint("FPFS", "norder", fallback=4)
        self.det_nrot = cparser.getint("FPFS", "det_nrot", fallback=4)
        self.pthres = cparser.getfloat("FPFS", "pthres", fallback=0.12)
        self.c0 = cparser.getfloat("FPFS", "c0")
        self.snr_min = cparser.getfloat("FPFS", "snr_min", fallback=10.0)
        self.r2_min = cparser.getfloat("FPFS", "r2_min", fallback=0.05)
        self.ename = cparser.get("FPFS", "ename", fallback="e1")
        assert int(self.ename[-1]) > 0
        self.egname = self.ename + "_g" + self.ename[-1]

        self.ncov_fname = cparser.get(
            "FPFS",
            "ncov_fname",
            fallback="",
        )
        if len(self.ncov_fname) == 0 or not os.path.isfile(self.ncov_fname):
            # estimate and write the noise covariance
            self.ncov_fname = os.path.join(self.cat_dir, "cov_matrix.fits")
        self.cov_matrix = Covariance.from_fits(self.ncov_fname)

        # shear setup
        self.shear_value = cparser.getfloat("simulation", "shear_value")

        # summary
        if not os.path.isdir(self.sum_dir):
            os.makedirs(self.sum_dir, exist_ok=True)
        self.test_obs = cparser.get("FPFS", "test_obs", fallback="snr_min")
        self.cut = getattr(self, self.test_obs)
        self.ofname = os.path.join(
            self.sum_dir,
            "bin_norder%d_%s_%02d.fits"
            % (
                self.norder,
                self.test_obs,
                int(self.cut * pf[self.test_obs]),
            ),
        )
        return

    def run(self, icore):
        id_range = self.get_range(icore)
        out = np.zeros((len(id_range), 4))
        ctask = CatalogTask(
            norder=self.norder,
            det_nrot=self.det_nrot,
            cov_matrix=self.cov_matrix,
        )
        ctask.update_parameters(
            snr_min=self.snr_min,
            r2_min=self.r2_min,
            c0=self.c0,
        )
        print("start core: %d, with id: %s" % (icore, id_range))
        start_time = time.time()
        assert self.cat_dir is not None
        en = self.ename
        egn = self.egname
        for icount, ifield in enumerate(id_range):
            for irot in range(self.nrot):
                nm1 = os.path.join(
                    self.cat_dir,
                    "src_1-%05d_%s-0_rot%d_%s.fits"
                    % (
                        ifield,
                        self.shear_comp_sim,
                        irot,
                        self.bands,
                    ),
                )
                src1 = Catalog.from_fits(nm1)
                mom1 = ctask.run(catalog=src1)
                nm2 = os.path.join(
                    self.cat_dir,
                    "src_1-%05d_%s-1_rot%d_%s.fits"
                    % (
                        ifield,
                        self.shear_comp_sim,
                        irot,
                        self.bands,
                    ),
                )
                src2 = Catalog.from_fits(nm2)
                mom2 = ctask.run(catalog=src2)

                e1m = np.sum(mom1[en] * mom1["w"])
                e1p = np.sum(mom2[en] * mom2["w"])
                r1m = np.sum(mom1[egn] * mom1["w"] + mom1[en] * mom1["w_g1"])
                r1p = np.sum(mom2[egn] * mom2["w"] + mom2[en] * mom2["w_g1"])

                out[icount, 0] = ifield
                out[icount, 1] = out[icount, 1] + (e1p - e1m)
                out[icount, 2] = out[icount, 2] + (e1m + e1p) / 2.0
                out[icount, 3] = out[icount, 3] + (r1m + r1p) / 2.0
                del src1, mom1, src2, mom2
        end_time = time.time()
        elapsed_time = (end_time - start_time) / 4.0
        print("elapsed time: %.2f seconds" % elapsed_time)
        return out

    def display_result(self, test_obs=None):
        if test_obs is None:
            cname = self.test_obs
        else:
            cname = test_obs

        spt = "bin_nord%d_%s_" % (self.norder, cname)
        flist = glob.glob("%s/%s*.fits" % (self.sum_dir, spt))
        res = []
        for fname in flist:
            obs = float(fname.split("/")[-1].split(spt)[-1].split(".fits")[0])
            obs = obs / float(pf[cname])
            print("%s is: %s" % (cname, obs))
            a = fitsio.read(fname)
            # rave = np.average(a[:, 3])
            # msk = (a[:, 3] >= 100) & (a[:, 3] < rave * 2.0)
            # a = a[msk]
            a = a[np.argsort(a[:, 0])]
            nsim = a.shape[0]
            b = np.average(a, axis=0)
            mbias = b[1] / b[3] / self.shear_value / 2.0 - 1
            print(
                "multiplicative bias:",
                mbias,
            )
            merr = (
                np.std(a[:, 1])
                / np.abs(np.average(a[:, 3]))
                / self.shear_value
                / 2.0
                / np.sqrt(nsim)
            )
            print(
                "1-sigma error:",
                merr,
            )
            cbias = b[2] / b[3]
            print("additive bias:", cbias)
            cerr = np.std(a[:, 2]) / np.abs(np.average(a[:, 3])) / np.sqrt(nsim)
            print(
                "1-sigma error:",
                cerr,
            )
            res.append((obs, mbias, merr, cbias, cerr))
        dtype = [
            (cname, "float"),
            ("mbias", "float"),
            ("merr", "float"),
            ("cbias", "float"),
            ("cerr", "float"),
        ]
        res = np.sort(np.array(res, dtype=dtype), order=cname)
        return res
