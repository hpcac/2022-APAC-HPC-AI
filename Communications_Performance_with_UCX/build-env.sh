set -e

# Replace PROJECT_NAME with another project, if you want to store files
# under a different project's storage unit rather than your primary project,
# e.g., 'public'.
export PROJECT_NAME=${PROJECT}

# Create scratch directory
export SCRATCH=/scratch/${PROJECT_NAME}/${USER}
if [ ! -d $SCRATCH ]; then
    mkdir $SCRATCH
fi

# Install miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ${SCRATCH}/miniconda.sh
bash ${SCRATCH}/miniconda.sh -b -p ${SCRATCH}/miniconda

# Activate miniconda
source $SCRATCH/miniconda/etc/profile.d/conda.sh
conda init

# Install mamba
conda activate base
conda install -y -c conda-forge mamba

# Create and activate UCX conda environment
# Make sure to accept the `nvcomp` license when asked
mamba env create -n ucx -f ./ucx-env.yml
conda activate ucx

# Load CUDA module
module load cuda/11.4.1

# Install UCX from source
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

make -j install
