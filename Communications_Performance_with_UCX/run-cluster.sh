CFG_FILE=${PBS_O_WORKDIR}/cluster.cfg
if [ ! -f $DIRECTORY_NAME ]; then
    echo "Config file ${CFG_FILE} not found. Make sure you're submitting the "
    echo "job from the directory containing launch.sh and cluster.cfg files."
fi

. ${CFG_FILE}

# Load CUDA module
module load cuda/11.4.1

SERVER_PROC=$1
NODE_NUM=$2

if [ "$SERVER_PROC" = true ]; then
    LOG_FILE=${JOB_DIR}/server-${HOSTNAME}-${PBS_JOBID}.log
    OUT_FILE=${JOB_DIR}/server-${HOSTNAME}-${PBS_JOBID}.out
else
    LOG_FILE=${JOB_DIR}/worker-${HOSTNAME}-${PBS_JOBID}.log
    OUT_FILE=${JOB_DIR}/worker-${HOSTNAME}-${PBS_JOBID}.out
fi

SERVER_FILE=${JOB_DIR}/.server-${PBS_JOBID}.json

cd $TEST_DIR

echo "hostname: $(hostname)" >> ${LOG_FILE}
echo "PWD: ${PWD}" >> ${LOG_FILE}

nvidia-smi >> ${LOG_FILE}
nvidia-smi topo -m >> ${LOG_FILE}

# Setup conda
source ${SCRATCH}/miniconda/etc/profile.d/conda.sh
conda activate ucx
echo "CONDA_PREFIX: ${CONDA_PREFIX}" >> ${LOG_FILE}

echo "TMPDIR: ${TMPDIR}" >> ${LOG_FILE}
echo "PBS_JOBID: ${PBS_JOBID}" >> ${LOG_FILE}
echo "PBS_NGPUS: ${PBS_NGPUS}" >> ${LOG_FILE}
echo "PBS_O_WORKDIR: ${PBS_O_WORKDIR}" >> ${LOG_FILE}

echo "SERVER_PROC: ${SERVER_PROC}" >> ${LOG_FILE}
echo "SERVER_FILE: ${SERVER_FILE}" >> ${LOG_FILE}
echo "NUM_ITERATIONS: ${NUM_ITERATIONS}" >> ${LOG_FILE}
echo "CHUNK_SIZE: ${CHUNK_SIZE}" >> ${LOG_FILE}
echo "NODE_NUM: ${NODE_NUM}" >> ${LOG_FILE}

if [ "$SERVER_PROC" = true ]; then
    echo "$ python cudf-merge.py \
        --server \
        --server-file ${SERVER_FILE} \
        --n-devs-on-net ${PBS_NGPUS} \
        --iter ${NUM_ITERATIONS} \
        -c ${CHUNK_SIZE}" &>> ${OUT_FILE}
    python cudf-merge.py \
        --server \
        --server-file ${SERVER_FILE} \
        --n-devs-on-net ${PBS_NGPUS} \
        --iter ${NUM_ITERATIONS} \
        -c ${CHUNK_SIZE} &>> ${OUT_FILE}

    # Clean server file
    rm -f ${SERVER_FILE}
else
    # Wait for server to come up
    while [ ! -f ${SERVER_FILE} ]; do
        sleep 3
    done
    sleep 3

    if [ "$WRITE_BASELINE" = true ]; then
        RESULTS_ARGS="--write-results-to-disk ${BASELINE_DIR}"
    fi
    if [ "$VERIFY_RESULTS" = true ]; then
        RESULTS_ARGS="--verify-results ${BASELINE_DIR}"
    fi

    echo "$ python cudf-merge.py \
        --devs "0,1,2,3" \
        --rmm-init-pool-size 30000000000 \
        --server-file ${SERVER_FILE} \
        --n-devs-on-net ${PBS_NGPUS} \
        --node-num ${NODE_NUM} \
        --iter ${NUM_ITERATIONS} \
        -c ${CHUNK_SIZE} \
        ${RESULTS_ARGS}" &>> ${OUT_FILE} &
    python cudf-merge.py \
        --devs "0,1,2,3" \
        --rmm-init-pool-size 30000000000 \
        --server-file ${SERVER_FILE} \
        --n-devs-on-net ${PBS_NGPUS} \
        --node-num ${NODE_NUM} \
        --iter ${NUM_ITERATIONS} \
        -c ${CHUNK_SIZE} \
        ${RESULTS_ARGS} &>> ${OUT_FILE} &
fi

echo "$HOSTNAME completed at: $(date)" >> ${LOG_FILE}
