# Dataframe Merge

A dataframe is a data format storing information in columnar fashion, that is, one or more columns can form a dataframe. In Python, the dataframe format has been made popular by [pandas](https://pandas.pydata.org/), a library that is capable of performing dozens of operations on dataframes to do what is often know as ETL (acronym for Extract, Transform, Load) of the data, that may be later fed to other data processing libraries for machine learning, deep learning, graph computing, visualization, etc.

Dataframes can be accelerated using [RAPIDS](https://rapids.ai/), more specifically with [cuDF](https://docs.rapids.ai/api/cudf/stable/), a library providing pandas-compatible API for GPU computing. To optimize data transfer, it is possible to utilize specialized interconnects such as NVIDIA NVLink and InfiniBand via UCX and UCX-Py (a Python library providing a layer for UCX communication), the latter introduces the capability to transfer GPU data, such as cuDF dataframes, directly from Python.

Because data often comes from a variety of different sources, merging dataframes lives at the core of ETL and is analogous to a join operation in SQL. The merge operation consists in combining dataframes based on the values of one or more columns (or in this context they are called "keys"), and if they match a specific rule, they will be merged and form a new dataframe. The rules may be:

- Use the keys of the left frame only (left merge);
- Use the keys of the right frame only (right merge);
- Use the union of keys from both left and right frames (outer merge);
- Use the intersection of keys from both left and right frames (inner merge);
- Use the cartesian product of keys from both left and right frames (cross merge).

Besides dataframe merge being a common operation, and despite its seemingly simple logic, it is a very communication-intensive operation, and thus it makes a great example for distributed computing with focus on communication performance.

# The Problem

The problem provided here consists of Python code that creates a central server to synchronize various workers. Each worker is a process running cuDF code on a single GPU in a Distributed setup. The code provided will in simple terms do the following:

- Launch a cluster with one or more GPU-capable nodes;
- Start one process that each will drive one of the GPUs (from here on referred to as "workers");
- Create pairs of UCX endpoints that will allow each GPU to directly communicate with one another, establishing a fully-connected network;
- Each worker will create and store two cuDF dataframes (left and right) in GPU memory;
  - The dataframes can be thought of as a chunk of one large dataframe distributed across the entirety of the cluster;
  - Each dataframe is produced from random data that is reproducible across iterations by reusing the same random seed, providing predictability of performance and results;
- Each left chunk will be merged to every right chunk;
- Results may be written to disk;
- Results may be verified for correctness against previously-generated data from the baseline implementation;
- Final performance results will be stored to disk once the cluster completes execution.

This problem is essentially the communication-heavy piece of Distributed frameworks such as [Dask](https://www.dask.org/). Performance improvements achieved in this problem are candidates for improving performance in such more complex frameworks.

# NCI/Gadi instructions

## Create development environment

**WARNING: This procedure will require at least 10GB of disk space.**

The steps to create the development environment are also available in the `build-env.sh` script, make sure you first adjust `PROJECT_NAME` in the script if needed. You will still be asked to accept the nvcomp license when installing with the script, make sure you follow the instructions to do so when asked.

### Create scratch directory

```bash
export PROJECT_NAME=${PROJECT}  # may be replaced by a different project, e.g., 'public'
export SCRATCH=/scratch/${PROJECT_NAME}/${USER}
mkdir $SCRATCH
```

### Install miniconda

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ${SCRATCH}/miniconda.sh
bash ${SCRATCH}/miniconda.sh -b -p ${SCRATCH}/miniconda
```

### Activate miniconda

```bash
source $SCRATCH/miniconda/etc/profile.d/conda.sh
conda init
```

Note that `conda init` will modify the user's `.bashrc` file, so it should not be necessary to run this step later again.

#### Install mamba to solve conda environment quickly

```
conda activate base
conda install -y -c conda-forge mamba
```

### Create and activate Dask conda environment

```bash
# Make sure to accept the `nvcomp` license when asked
mamba env create -n ucx -f ./ucx-env.yml
mamba activate ucx
```

### Load CUDA module (required for building UCX)

```bash
module load cuda/11.4.1
```

### Install UCX From Source

This step is only required to enable InfiniBand, UCX conda packages don't currently include support for that.

```bash
mkdir -p ${SCRATCH}/src
cd ${SCRATCH}/src
git clone https://github.com/openucx/ucx
cd ucx
git checkout v1.12.1
./autogen.sh
mkdir build
cd build

../contrib/configure-release --prefix=$CONDA_PREFIX --with-cuda=$CUDA_HOME --enable-mt
# Alternatively as below to enable UCX debug build
# ../contrib/configure-devel --prefix=$CONDA_PREFIX --with-cuda=$CUDA_HOME --enable-mt

# Use only 8 threads to be mindful of other users on the login node. If running
# on a compute node, `make -j install` can be used instead.
make -j8 install
```

### Resume work

After you log out and want to resume, it should suffice to simply re-activate the `ucx` environment:

```
conda activate ucx
```

## Preparations to run

### Create password-less SSH key pair

The job we will run uses SSH to setup a cluster. This will require that your user has permission to login to remote compute nodes without a password.

```
ssh-keygen -P "" -f ~/.ssh/id_rsa
cat ~/.ssh/id_rsa.pub >> ~/.ssh/authorized_keys
```

### Configuration

Before running, it is necessary to change a few of the configurations. Please be sure to read the various different configurations in `cluster.cfg`, the documentation in that file should be self-explanatory. Specifically, you may need to adjust `PROJECT_NAME`, as well as the "Problem parameters" section in that file.

Each compute job is submitted using `launch.sh`. Because PBS expects parameters to the job to be passed as comments at the top of the launch script, you will also need to adjust them in `launch.sh`. In particular, you will need to adjust `ngpus`, `ncpus` and `mem` if you want to change the amount of nodes on which you will launch the job. By default it is configured for 16 GPUs (4 nodes), and all three options must be set accordingly to request compute resources. Make sure to check Gadi's [Queue Limits](https://opus.nci.org.au/display/Help/Queue+Limits) and [Queue Structure](https://opus.nci.org.au/display/Help/Queue+Structure) documentation for details on changing the number of nodes/GPUs.

Still in `launch.sh` you will also need to change `storage` such that your scratch directory will be available in the compute nodes. By default, it will have `/scratch/${PROJECT}` mounted, where `PROJECT` is the name of the primary project your user is assigned to, which can be verified running `nci_account` on your shell. Additionally, you will need to adjust `PROJECT_NAME` in `cluster.cfg` to match the same scratch directory you setup in `launch.sh`. Finally, they must all meet the scratch directory you used when setting up your conda environment.

## Running

### Launching the job

**WARNING: To launch the job, you must first `cd` into the directory containing `cluster.cfg`, `launch.sh`, `run-cluster.sh` and `cudf-merge.py` files. This is paramount as the `${PBS_O_WORKDIR}` environment variable that is automatically defined by PBS will be used by the scripts to find other files.**

Launching the job is then very simple, and can be done from shell using the following command:

```
$ qsub launch.sh
```

### Job output

Once your job completes, you will have output and logs of each of the `cudf-merge.py` processes (server and workers) in `${SCRATCH}/job-out`. Normally, you will only be interested in `${SCRATCH}/job-out/server-*.out`. This file contains the performance results of your cluster, and should look similar to the example below for the large dataset (25 million rows per chunk):

```
cuDF merge benchmark
--------------------------------------------------------------------------------------------------------------
Number of devices         | 16
Rows per chunk            | 25000000
Total data processed      | 1.16 TiB
Data processed per iter   | 11.92 GiB
==============================================================================================================
Wall-clock                | 254.03 s
Bandwidth                 | 433.89 MiB/s
Throughput                | 4.69 GiB/s
==============================================================================================================
Iteration                 | Wall-clock                | Bandwidth                 | Throughput
0                         | 2.63 s                    | 438.14 MiB/s              | 4.53 GiB/s
1                         | 2.28 s                    | 460.18 MiB/s              | 5.23 GiB/s
2                         | 2.69 s                    | 424.37 MiB/s              | 4.43 GiB/s
3                         | 2.65 s                    | 409.93 MiB/s              | 4.50 GiB/s
4                         | 2.57 s                    | 429.17 MiB/s              | 4.63 GiB/s
5                         | 2.61 s                    | 427.51 MiB/s              | 4.57 GiB/s
6                         | 2.51 s                    | 438.80 MiB/s              | 4.74 GiB/s
7                         | 2.53 s                    | 428.98 MiB/s              | 4.71 GiB/s
8                         | 2.59 s                    | 426.56 MiB/s              | 4.60 GiB/s
9                         | 2.43 s                    | 455.33 MiB/s              | 4.91 GiB/s
...
```

The remaining files inside `${SCRATCH}/job-out` should be checked in the event that something went wrong.

### Baseline and result correctness

The first time you run, you will want to write the baseline results. This can be achieved by setting `WRITE_BASELINE=true` in `cluster.cfg`. After you have done this once, you can set it to `WRITE_BASELINE=false` to avoid overwriting the baseline when you make any configuration or code modifications. This result is always reproducible, and can be verified with `VERIFY_RESULTS=true` to ensure correctness after the baseline was generated.

When you run your cluster with results verification turned on, each worker (i.e., rank) will write the results for each chunk and each iteration under:

```
${SCRATCH}/baseline/verify-logs/results_${ROWS_PER_CHUNK}_${RANK}_${ITERATION}.log
```

Each of the files above should read "OK" if no correctness errors were introduced. If any result is incorrect, the file corresponding to rows per chunk, rank and iteration will present the output of the error and may used for manual inspection.

No guarantees are made as to performance remaining state when `VERIFY_RESULTS=true`, therefore you may want to disable that while you're working on improving performance.

Performance and correctness will be evaluated individually to ensure fairness as to performance results. In other words, you can rest assured correctness checking will not degrade the perfomance of your results.

Only one submission per candidate/team will be accepted and the final submission must ensure results are indeed correct. If results are incorrect, the submission will be disqualified. This is necessary to ensure no undefined behavior or errors to the computation were inadvertently introduced.

# Objectives and Rules

The objective is for the user to improve merge benchmark's overall bandwidth (how fast data can be transferred) and throughput (how fast data can be processed _and_ transferred).

To achieve the expected results mentioned above, the user may modify cuDF, UCX, UCX-Py, and any other open-source library on the stack, both their configuration and source code is allowed.

For configuration changes, the user is expected to submit patches to the provided scripts (e.g., `launch.sh` and `run-cluster.sh`) that can be executed in the same way as the original code is provided. If necessary, any required instructions must be submitted as well.

If any code changes are made, the user must submit any patches and scripts for an automated build of the stack. For example, the original conda environment can be used as base, and the user-submitted scripts will build and install software overwriting the original packages in the conda environment.

The user is not allowed to upgrade nor downgrade any libraries used. However, the user is allowed to back- and forward-port modifications of any of the respective libraries, if that modification is judged beneficial for performance by the user.

The user is not allowed to make any changes to the problem itself. Any changes preserving equality of the input data generation, the problem's output and final presentation of results is allowed. If the final submission modifies input or output data, as well as how results are presented as an output of the provided code, the submission will be disqualified.

At the end, the results will be executed in Gadi and graded based on best bandwidth and throughput performance achieved.

## Problem parameters

Two subsets of the problem will be evaluated:

1. Small (1 million rows, `1_000_000`), 100 iterations, 16 workers (4 Gadi full nodes); and
1. Large (25 million rows, `25_000_000`), 100 iterations, 16 workers (4 Gadi full nodes).

Different changes may improve or degrade performance for different problem sizes. Therefore, it's important for the user to attempt improving performance while keeping in mind robustness of the system.

To prevent overfitting for a specific case, it is required that the submissions ensures best performance for both small and large problems, therefore the user must make sure one set of parameters and software can demonstrate good performance for both small and large problems. In other words, compiling a different set of software for small and large problems is not allowed, and configurations/parameters that are used to run the small problem must be used as well for the large problem, and vice-versa. If changes are required to work on a threshold, the user must ensure the threshold is handled programatically and not via different software, parameters or configurations that must be defined before launching the application.

## Marking criteria

- Report: 40%;
- Small problem performance: 30%;
- Large problem performance: 30%.
