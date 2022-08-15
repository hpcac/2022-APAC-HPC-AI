#PBS -P il82
#PBS -N cudf-merge-benchmark
#PBS -q gpuvolta
#PBS -l walltime=00:10:00
#PBS -l ngpus=16
#PBS -l ncpus=192
#PBS -l mem=1536GB
#PBS -l storage=scratch/il82

CFG_FILE=${PBS_O_WORKDIR}/cluster.cfg
if [ ! -f $DIRECTORY_NAME ]; then
    echo "Config file ${CFG_FILE} not found. Make sure you're submitting the "
    echo "job from the directory containing launch.sh and cluster.cfg files."
fi

. ${CFG_FILE}

################################################################################
#         Launch server and workers -- shouldn't be necessary to change        #
################################################################################
# Load CUDA module
# module load cuda/11.4.1

export ALL_HOSTS=$(cat $PBS_NODEFILE | sort | uniq)

make_dir() {
   DIRECTORY_NAME=$1
   if [ -z ${DIRECTORY_NAME} ]; then
       echo "Empty directory name, this probably means JOB_DIR is undefined"
       exit 1
   fi

   if [ ! -d ${DIRECTORY_NAME} ]; then
       mkdir -p ${DIRECTORY_NAME}
   fi
}

make_dir $JOB_DIR

echo "PWD: $PWD"
echo "SCHEDULER_HOST: ${SCHEDULER_HOST}"
echo "ALL_HOSTS: ${ALL_HOSTS}"

cd $TEST_DIR

echo "hostname: $(hostname)"
nvidia-smi
nvidia-smi topo -m

# Setup conda
source ${SCRATCH}/miniconda/etc/profile.d/conda.sh
conda activate ucx
echo $CONDA_PREFIX

python -c "import ucp; print(ucp.get_ucx_version())"
ucx_info -v

# Launch workers
NODE_NUM=0
for host in $ALL_HOSTS; do
    echo "ssh -o StrictHostKeyChecking=no -n -f $host \
        \"cd $TEST_DIR; \
        TMPDIR=${TMPDIR} PBS_JOBID=${PBS_JOBID} PBS_NGPUS=${PBS_NGPUS} PBS_O_WORKDIR=${PBS_O_WORKDIR} nohup bash $TEST_DIR/run-cluster.sh \
        false \
        ${NODE_NUM} &> /dev/null &\""
    ssh -o StrictHostKeyChecking=no -n -f $host \
        "cd $TEST_DIR; \
        TMPDIR=${TMPDIR} PBS_JOBID=${PBS_JOBID} PBS_NGPUS=${PBS_NGPUS} PBS_O_WORKDIR=${PBS_O_WORKDIR} nohup bash $TEST_DIR/run-cluster.sh \
        false \
        ${NODE_NUM} &> /dev/null &"

    NODE_NUM=$(expr ${NODE_NUM} + 1)
done

# Launch server
echo "bash $TEST_DIR/run-cluster.sh true &> /dev/null"
bash $TEST_DIR/run-cluster.sh true

echo "Completed at $(date)"
