# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
# See file LICENSE for terms.

"""
Benchmark send receive on one machine
"""
import argparse
import asyncio
import cProfile
import gc
import io
import os
import pickle
import pstats
from time import monotonic as clock

import cupy
import numpy as np

import ucp

from utils import (
    format_bytes,
    format_time,
    hmean,
    print_multi,
    print_separator,
)
from utils_multi_node import (
    run_on_multiple_nodes_server,
    run_on_multiple_nodes_worker,
)

# Must be set _before_ importing RAPIDS libraries (cuDF, RMM)
os.environ["RAPIDS_NO_INITIALIZE"] = "True"


import cudf  # noqa
import rmm  # noqa


def sizeof_cudf_dataframe(df):
    return int(
        sum(col.memory_usage for col in df._data.columns) + df._index.memory_usage()
    )


def write_chunk(path, df, i_chunk, local_size, chunk_type):
    os.makedirs(path, exist_ok=True)
    fname = f"{path}/{chunk_type}_{local_size}_{i_chunk}.parquet"
    df.to_parquet(fname)


async def send_df(ep, df):
    header, frames = df.serialize()
    header["frame_ifaces"] = [f.__cuda_array_interface__ for f in frames]
    header = pickle.dumps(header)
    header_nbytes = np.array([len(header)], dtype=np.uint64)
    await ep.send(header_nbytes)
    await ep.send(header)
    for frame in frames:
        await ep.send(frame)


async def recv_df(ep):
    header_nbytes = np.empty((1,), dtype=np.uint64)
    await ep.recv(header_nbytes)
    header = bytearray(header_nbytes[0])
    await ep.recv(header)
    header = pickle.loads(header)

    frames = [
        cupy.empty(iface["shape"], dtype=iface["typestr"])
        for iface in header["frame_ifaces"]
    ]
    for frame in frames:
        await ep.recv(frame)

    cudf_typ = pickle.loads(header["type-serialized"])
    return cudf_typ.deserialize(header, frames)


async def barrier(rank, eps):
    if rank == 0:
        await asyncio.gather(*[ep.recv(np.empty(1, dtype="u1")) for ep in eps.values()])
    else:
        await eps[0].send(np.zeros(1, dtype="u1"))


async def send_bins(eps, bins):
    futures = []
    for rank, ep in eps.items():
        futures.append(send_df(ep, bins[rank]))
    await asyncio.gather(*futures)


async def recv_bins(eps, bins):
    futures = []
    for ep in eps.values():
        futures.append(recv_df(ep))
    bins.extend(await asyncio.gather(*futures))


async def exchange_and_concat_bins(rank, eps, bins, timings=None):
    ret = [bins[rank]]
    if timings is not None:
        t1 = clock()
    await asyncio.gather(recv_bins(eps, ret), send_bins(eps, bins))
    if timings is not None:
        t2 = clock()
        timings.append(
            (
                t2 - t1,
                sum(
                    [sizeof_cudf_dataframe(b) for i, b in enumerate(bins) if i != rank]
                ),
            )
        )
    return cudf.concat(ret)


async def distributed_join(args, rank, eps, left_table, right_table, timings=None):
    left_bins = left_table.partition_by_hash(["key"], args.n_chunks)
    right_bins = right_table.partition_by_hash(["key"], args.n_chunks)

    left_df = await exchange_and_concat_bins(rank, eps, left_bins, timings)
    right_df = await exchange_and_concat_bins(rank, eps, right_bins, timings)
    return left_df.merge(right_df, on="key")


def generate_chunk(i_chunk, local_size, num_chunks, chunk_type, frac_match):
    seed = num_chunks * int(chunk_type == "left") + i_chunk
    cupy.random.seed(seed)

    start = local_size * i_chunk
    stop = start + local_size

    df = cudf.DataFrame(
        {
            "key": cupy.random.randint(0, 100, size=local_size*num_chunks, dtype="int64")[start:stop],
            "payload": cupy.arange(local_size, dtype="int64"),
        }
    )
    if chunk_type == "left":
        # Left dataframe
        #
        # "key" column is a unique sample within [0, local_size * num_chunks)
        #
        # "shuffle" column is a random selection of partitions (used for shuffle)
        #
        # "payload" column is a random permutation of the chunk_size

        start = local_size * i_chunk
        stop = start + local_size

        df = cudf.DataFrame(
            {
                "key": cupy.arange(start, stop=stop, dtype="int64"),
                "payload": cupy.arange(local_size, dtype="int64"),
            }
        )
    else:
        # Right dataframe
        #
        # "key" column matches values from the build dataframe
        # for a fraction (`frac_match`) of the entries. The matching
        # entries are perfectly balanced across each partition of the
        # "base" dataframe.
        #
        # "payload" column is a random permutation of the chunk_size

        # Step 1. Choose values that DO match
        sub_local_size = local_size // num_chunks
        sub_local_size_use = max(int(sub_local_size * frac_match), 1)
        arrays = []
        for i in range(num_chunks):
            bgn = (local_size * i) + (sub_local_size * i_chunk)
            end = bgn + sub_local_size
            ar = cupy.arange(bgn, stop=end, dtype="int64")
            arrays.append(cupy.random.permutation(ar)[:sub_local_size_use])
        key_array_match = cupy.concatenate(tuple(arrays), axis=0)

        # Step 2. Add values that DON'T match
        missing_size = local_size - key_array_match.shape[0]
        start = local_size * num_chunks + local_size * i_chunk
        stop = start + missing_size
        key_array_no_match = cupy.arange(start, stop=stop, dtype="int64")

        # Step 3. Combine and create the final dataframe chunk
        key_array_combine = cupy.concatenate(
            (key_array_match, key_array_no_match), axis=0
        )
        df = cudf.DataFrame(
            {
                "key": cupy.random.permutation(key_array_combine),
                "payload": cupy.arange(local_size, dtype="int64"),
            }
        )
    return df


def verify_results(path, test_df, i_chunk, local_size, chunk_type, iteration):
    fname = f"{path}/{chunk_type}_{local_size}_{i_chunk}.parquet"
    base_df = cudf.read_parquet(fname)

    base_df = base_df.sort_values(by=["key", "payload_x", "payload_y"], ignore_index=True)
    test_df = test_df.sort_values(by=["key", "payload_x", "payload_y"], ignore_index=True)

    path = path + "/verify-logs"
    os.makedirs(path, exist_ok=True)
    log_fname = f"{path}/{chunk_type}_{local_size}_{i_chunk}_{iteration}.log"
    log_msg = "OK"

    try:
        cudf.testing.assert_frame_equal(test_df, base_df)
    except AssertionError as e:
        print(
            f"Result verification for rank {i_chunk} failed in iteration "
            f"{iteration}, full results in {log_fname}."
        )
        log_msg = str(e)

    with open(log_fname, "w") as f:
        f.write(log_msg)


async def worker(rank, eps, args):
    # Setting current device and make RMM use it
    rmm.reinitialize(pool_allocator=True, initial_pool_size=args.rmm_init_pool_size)

    # Make cupy use RMM
    cupy.cuda.set_allocator(rmm.rmm_cupy_allocator)

    frac_match = 0.3

    left_df = generate_chunk(rank, args.chunk_size, args.n_chunks, "left", frac_match)
    right_df = generate_chunk(rank, args.chunk_size, args.n_chunks, "right", frac_match)

    # Let's warmup and sync before benchmarking
    for i in range(args.warmup_iter):
        await distributed_join(args, rank, eps, left_df, right_df)
        await barrier(rank, eps)

    if args.cuda_profile:
        cupy.cuda.profiler.start()

    if args.profile:
        pr = cProfile.Profile()
        pr.enable()

    iter_results = {"bw": [], "wallclock": [], "throughput": [], "data_processed": []}
    timings = []
    t1 = clock()
    for i in range(args.iter):
        iter_timings = []

        iter_t = clock()
        result_df = await distributed_join(args, rank, eps, left_df, right_df, iter_timings)
        await barrier(rank, eps)
        iter_took = clock() - iter_t

        # Ensure the number of matches falls within `args.frac_match` +/- 1%
        expected_len = args.chunk_size * frac_match
        expected_len_err = expected_len * 0.01
        assert len(result_df) in range(
            int(expected_len - expected_len_err), int(expected_len + expected_len_err)
        )

        if args.write_results_to_disk and i == 0:
            write_chunk(args.write_results_to_disk, result_df, rank, args.chunk_size, "results")

        if args.verify_results:
            verify_results(args.verify_results, result_df, rank, args.chunk_size, "results", i)

        # Avoid cyclic references preventing garbage collection
        del result_df

        iter_bw = sum(t[1] for t in iter_timings) / sum(t[0] for t in iter_timings)
        iter_data_processed = len(left_df) * sum([t.itemsize for t in left_df.dtypes])
        iter_data_processed += len(right_df) * sum([t.itemsize for t in right_df.dtypes])
        iter_throughput = args.n_chunks * iter_data_processed / iter_took

        iter_results["bw"].append(iter_bw)
        iter_results["wallclock"].append(iter_took)
        iter_results["throughput"].append(iter_throughput)
        iter_results["data_processed"].append(iter_data_processed)

        timings += iter_timings

    took = clock() - t1

    if args.profile:
        pr.disable()
        s = io.StringIO()
        ps = pstats.Stats(pr, stream=s)
        ps.dump_stats("%s.%0d" % (args.profile, rank))

    if args.cuda_profile:
        cupy.cuda.profiler.stop()

    data_processed = len(left_df) * sum([t.itemsize * args.iter for t in left_df.dtypes])
    data_processed += len(right_df) * sum([t.itemsize * args.iter for t in right_df.dtypes])

    return {
        "bw": sum(t[1] for t in timings) / sum(t[0] for t in timings),
        "wallclock": took,
        "throughput": args.n_chunks * data_processed / took,
        "data_processed": data_processed,
        "iter_results": iter_results,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "-d",
        "--devs",
        metavar="LIST",
        default="0",
        type=str,
        help='GPU devices to use (default "0").',
    )
    parser.add_argument("-c", "--chunk-size", type=int, default=4, metavar="N")
    parser.add_argument(
        "--profile",
        metavar="FILENAME",
        default=None,
        type=str,
        help="Write profile for each worker to `filename.RANK`",
    )
    parser.add_argument(
        "--cuda-profile",
        default=False,
        action="store_true",
        help="Enable CUDA profiling, use with `nvprof --profile-child-processes \
                --profile-from-start off`",
    )
    parser.add_argument(
        "--rmm-init-pool-size",
        metavar="BYTES",
        default=None,
        type=int,
        help="Initial RMM pool size (default  1/2 total GPU memory)",
    )
    parser.add_argument(
        "--iter",
        default=1,
        type=int,
        help="Number of benchmark iterations.",
    )
    parser.add_argument(
        "--warmup-iter",
        default=5,
        type=int,
        help="Number of warmup iterations.",
    )
    parser.add_argument(
        "--server",
        default=False,
        action="store_true",
        help="Run server only.",
    )
    parser.add_argument(
        "--server-file",
        type=str,
        help="File to store server's address (if `--server` is specified) or to "
        "read its address from otherwise.",
    )
    parser.add_argument(
        "--n-devs-on-net",
        type=int,
        help="Number of devices in the entire network.",
    )
    parser.add_argument(
        "--node-num",
        type=int,
        help="On a multi-node setup, specify the number of the node that this "
        "process is running. Must be a unique number in the "
        "[0, `--n-workers` / `len(--devs)`) range.",
    )
    parser.add_argument(
        "--write-results-to-disk",
        type=str,
        help="Write generated results to the specified path, which may later be "
        "used to verify correctness.",
    )
    parser.add_argument(
        "--verify-results",
        type=str,
        help="Verify that results are correct with previously-generated results "
        "stored on the specified path.",
    )
    args = parser.parse_args()

    if args.server_file is None:
        raise RuntimeError(
            "`--server-file` needs to be specified with a path reachable by all nodes."
        )

    if args.n_devs_on_net is None:
        raise RuntimeError(
            "The total number of devices in the network must be specified with "
            "`--n-devs-on-net`."
        )
    elif args.n_devs_on_net < 2:
        raise RuntimeError(
            "The total number of devices in the network specified with "
            "`--n-devs-on-net` must be at least 2.",
        )

    if not args.server and args.node_num is None:
        raise RuntimeError(
            "Each worker on a multi-node is required to specify `--node-num`."
        )

    args.devs = [int(d) for d in args.devs.split(",")]
    args.n_chunks = args.n_devs_on_net
    args.node_n_workers = len(args.devs)

    return args


def main():
    args = parse_args()
    if not args.server:
        ranks = range(args.n_chunks)
        assert len(ranks) > 1
        assert len(ranks) % 2 == 0

    if args.server:
        stats = run_on_multiple_nodes_server(
            args.server_file,
            args.n_devs_on_net,
        )
    elif args.server_file:
        run_on_multiple_nodes_worker(
            args.server_file,
            args.n_chunks,
            args.node_n_workers,
            args.node_num,
            worker,
            worker_args=args,
            ensure_cuda_device=True,
        )
        return

    wc = stats[0]["wallclock"]
    bw = hmean(np.array([s["bw"] for s in stats]))
    tp = stats[0]["throughput"]
    dp = sum(s["data_processed"] for s in stats)
    dp_iter = sum(s["iter_results"]["data_processed"][0] for s in stats)

    print("cuDF merge benchmark")
    print_separator(separator="-", length=110)
    print_multi(values=["Number of devices", f"{args.n_devs_on_net}"])
    print_multi(values=["Rows per chunk", f"{args.chunk_size}"])
    print_multi(values=["Total data processed", f"{format_bytes(dp)}"])
    print_multi(values=["Data processed per iter", f"{format_bytes(dp_iter)}"])
    print_separator(separator="=", length=110)
    print_multi(values=["Wall-clock", f"{format_time(wc)}"])
    print_multi(values=["Bandwidth", f"{format_bytes(bw)}/s"])
    print_multi(values=["Throughput", f"{format_bytes(tp)}/s"])
    print_separator(separator="=", length=110)
    print_multi(values=["Iteration", "Wall-clock", "Bandwidth", "Throughput"])
    for i in range(args.iter):
        iter_results = stats[0]["iter_results"]

        iter_wc = iter_results["wallclock"][i]
        iter_bw = hmean(np.array([s["iter_results"]["bw"][i] for s in stats]))
        iter_tp = iter_results["throughput"][i]

        print_multi(
            values=[
                i,
                f"{format_time(iter_wc)}",
                f"{format_bytes(iter_bw)}/s",
                f"{format_bytes(iter_tp)}/s",
            ]
        )


if __name__ == "__main__":
    main()
