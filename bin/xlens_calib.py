#!/usr/bin/env python
import os
import sys
from argparse import ArgumentParser

import numpy as np
import schwimmbad

# from memory_profiler import profile

task_list = [
    "simulate_image",
    "prepare_star",
    "measure_dm",
    "measure_fpfs",
    "summary_fpfs",
    "neff_fpfs",
]


def get_processor_count(pool, args):
    if isinstance(pool, schwimmbad.MPIPool):
        # MPIPool
        from mpi4py import MPI

        return MPI.COMM_WORLD.Get_size() - 1
    elif isinstance(pool, schwimmbad.MultiPool):
        # MultiPool
        return args.n_cores
    else:
        # SerialPool
        return 1


# @profile
def run(pool, cmd_args, taskname, min_id, max_id, ncores):
    from xlens.simulation.simulator.base import SimulateImage

    w0 = SimulateImage(cmd_args.config)
    if taskname.lower() == "simulate_image":
        input_list = list(range(min_id, max_id))
        for _ in pool.map(w0.run, input_list):
            pass
    elif taskname.lower() == "measure_dm":
        from xlens.simulation.measure import ProcessSimDM

        worker = ProcessSimDM(cmd_args.config)
        input_list = w0.get_sim_fnames(min_id=min_id, max_id=max_id)
        for _ in pool.map(worker.run, input_list):
            pass
    elif taskname.lower() == "prepare_star":
        from xlens.simulation.simulator.loader import MakeBrightInfo

        worker = MakeBrightInfo(cmd_args.config)
        input_list = w0.get_sim_fnames(
            min_id=min_id,
            max_id=max_id,
            field_only=True,
        )
        for _ in pool.map(worker.run, input_list):
            pass
    elif taskname.lower() == "measure_fpfs":
        from xlens.simulation.measure import ProcessSimFpfs

        # import fitsio

        worker = ProcessSimFpfs(cmd_args.config)
        input_list = w0.get_sim_fnames(min_id=min_id, max_id=max_id)
        for _ in pool.map(worker.run, input_list):
            pass
    elif taskname.lower() == "summary_fpfs":
        import fitsio

        from xlens.simulation.summary import SummarySimFpfs

        worker = SummarySimFpfs(
            cmd_args.config,
            min_id=min_id,
            max_id=max_id,
            ncores=ncores,
        )
        if not os.path.isfile(worker.ofname):
            olist = pool.map(worker.run, np.arange(ncores))
            fitsio.write(worker.ofname, np.vstack(list(olist)))
        worker.display_result()
    elif taskname.lower() == "neff_fpfs":
        from xlens.simulation.neff import NeffSimFpfs

        worker = NeffSimFpfs(
            cmd_args.config,
            min_id=min_id,
            max_id=max_id,
            ncores=ncores,
        )
        olist = pool.map(worker.run, np.arange(ncores))
        outcome = np.vstack(list(olist))
        std = np.std(outcome[:, 0]) / np.abs(np.average(outcome[:, 1]))
        print("std: %s" % std)
        neff = (0.26 / std) ** 2.0 / worker.area
        print("neff: %s" % neff)
    else:
        raise ValueError(
            "taskname cannot be set to %s, we only support %s"
            % (taskname, task_list)
        )
    pool.close()
    sys.exit(0)
    return


# @profile
def main():
    parser = ArgumentParser(description="simulate blended images")
    parser.add_argument(
        "task_name",
        type=str,
        help="The name fo the task to run.",
    )
    parser.add_argument(
        "--min_id",
        default=0,
        type=int,
        help="minimum simulation id number, e.g. 0",
    )
    parser.add_argument(
        "--max_id",
        default=5000,
        type=int,
        help="maximum simulation id number, e.g. 4000",
    )
    parser.add_argument(
        "--config",
        required=True,
        type=str,
        help="configuration file name",
    )
    #
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--ncores",
        dest="n_cores",
        default=1,
        type=int,
        help="Number of processes (uses multiprocessing).",
    )
    group.add_argument(
        "--mpi",
        dest="mpi",
        default=False,
        action="store_true",
        help="Run with MPI.",
    )
    cmd_args = parser.parse_args()
    min_id = cmd_args.min_id
    max_id = cmd_args.max_id
    pool = schwimmbad.choose_pool(mpi=cmd_args.mpi, processes=cmd_args.n_cores)
    ncores = get_processor_count(pool, cmd_args)
    taskname = cmd_args.task_name
    run(
        pool=pool,
        cmd_args=cmd_args,
        taskname=taskname,
        min_id=min_id,
        max_id=max_id,
        ncores=ncores,
    )
    return


if __name__ == "__main__":
    main()
