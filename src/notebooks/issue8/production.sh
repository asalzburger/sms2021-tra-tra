#!/bin/bash



####################################################################################################
###                                     variables/constants                                      ###
####################################################################################################


# particle specs
N_EVENTS=100
N_PARTICLES_PER_EVENT=25
PDG=0

# eta specs
MIN_ETA=-5.5
MAX_ETA=5.5
ETA=5.5
ETA_RANGE="${MIN_ETA}:${MAX_ETA}"

# momentum specs
MIN_MOM_GEV=0.5
MAX_MOM_GEV=10.0
MOM_RANGE="${MIN_MOM_GEV}:${MAX_MOM_GEV}"

# (constant) magnetic field specs
CONSTANT_BFIELD_X=0
CONSTANT_BFIELD_Y=0
CONSTANT_BFIELD_Z=2
CONSTANT_BFIELD="${CONSTANT_BFIELD_X}:${CONSTANT_BFIELD_Y}:${CONSTANT_BFIELD_Z}"

# directory and path constants
BINARIES_DIR="/home/andreas/work/libs/acts_build/bin"
DATA_DIR="/home/andreas/Desktop/CERN/tra-tra/data"
DD4HEP_INPUT="/home/andreas/work/libs/acts/thirdparty/OpenDataDetector/xml/OpenDataDetector.xml"
MAT_SUFFIX="-with-material-effects"
ODD_BFIELD_SUFFIX="-non-homogenous-magnetic-field"

# executable paths
PARTICLE_GUN_BIN="${BINARIES_DIR}/ActsExampleParticleGun"
FATRAS_SIMULATION_BIN="${BINARIES_DIR}/ActsExampleFatrasDD4hep"

# simulation specs
ODD_DATA_DIR="/home/andreas/Desktop/CERN/other-repos/OpenDataDetector/data"
MAT_INPUT_FILE="${ODD_DATA_DIR}/odd-material-map.root"
ODD_BFIELD_INPUT="${ODD_DATA_DIR}/odd-bfield.root"
USE_MAT=true
USE_ODD_BFIELD=true
ADDITIONAL_ARGS=""



####################################################################################################
###                                      utility functions                                       ###
####################################################################################################


run_acts_particle_gun()
{
    $PARTICLE_GUN_BIN                              \
        -n ${N_EVENTS}                             \
        --gen-nparticles ${N_PARTICLES_PER_EVENT}  \
        --gen-pdg ${PDG}                           \
        --gen-mom-gev ${MOM_RANGE}                 \
        --gen-mom-transverse true                  \
        --gen-eta ${ETA_RANGE}                     \
        --output-csv
}


add_additional_args()
{
    ADDITIONAL_ARGS=""

    # ideal case
    if [[ "${USE_MAT}" = false ]] && [[ "${USE_ODD_BFIELD}" = false ]]; then
        ADDITIONAL_ARGS="--bf-constant-tesla ${CONSTANT_BFIELD}"
    else
        # add material effect
        if [ "${USE_MAT}" = true ]; then
            ADDITIONAL_ARGS="--mat-input-type=file --mat-input-file=${MAT_INPUT_FILE}"
        fi
        # use a non-homogenous magnetic field B
        if [ "${USE_ODD_BFIELD}" = true ]; then
            ADDITIONAL_ARGS="${ADDITIONAL_ARGS} --bf-map-file=${ODD_BFIELD_INPUT}"
        fi
    fi
}


run_fatras_dd4hep_simulation()
{
    add_additional_args

    $FATRAS_SIMULATION_BIN            \
        --dd4hep-input=$DD4HEP_INPUT  \
        --output-csv                  \
        "$ADDITIONAL_ARGS"            \
        --input-dir=./
}


add_appropriate_suffix_to_dir()
{
    if [ "${USE_MAT}" = true ]; then
        DIR="${DIR}${MAT_SUFFIX}"
    fi
    if [ "${USE_ODD_BFIELD}" = true ]; then
        DIR="${DIR}${ODD_BFIELD_SUFFIX}"
    fi
}


move_dataset_to_corresponding_directory()
{
    MOM_RANGE_STR="${MIN_MOM_GEV}to${MAX_MOM_GEV}GeV"
    DIR="${DATA_DIR}/pdg${PDG}/pdg${PDG}-n${N_PARTICLES_PER_EVENT}-${MOM_RANGE_STR}-${ETA}eta"
    add_appropriate_suffix_to_dir

    rm -rf $DIR
    mkdir -p $DIR
    mv ./*.csv $DIR
    rm ./*.tsv
}


create_dataset()
{
    run_acts_particle_gun
    run_fatras_dd4hep_simulation
    move_dataset_to_corresponding_directory
}


create_all_datasets()
{
    # ideal dataset (without material effects and with homogenous magnetic field B)
    USE_MAT=false
    USE_ODD_BFIELD=false
    create_dataset

    # dataset with material effects (and homogenous magnetic field B)
    USE_MAT=true
    USE_ODD_BFIELD=false
    create_dataset

    # dataset with non-homogenous magnetic field (and without material effects)
    USE_MAT=false
    USE_ODD_BFIELD=true
    create_dataset

    # dataset with material effects and non-homogenous magnetic field
    USE_MAT=true
    USE_ODD_BFIELD=true
    create_dataset
}



####################################################################################################
###                                parse command line arguments                                  ###
####################################################################################################


# prompt to explain the input to the user
PROMPT="./production.sh pdg"

# check if user wants to see the prompt
if [[ $1 == "-h" || $1 == "--help" ]]; then
    echo -e "\n$PROMPT\n"
    exit 0
fi

# start parsing command line arguments
if [[ $# != 1 ]]; then
    echo -e "\nERROR: Incorrect number of command line arguments.\n\nUsage: ${PROMPT}\n"
    exit 1
fi

# parse the particle data group (pdg)
if [[ $1 -le 0 ]]; then
    echo -e "\nERROR: Unknown pdg $1.\n"
    exit 2
fi
PDG=$1



####################################################################################################
###                                   main driver mechanism                                      ###
####################################################################################################


create_all_datasets
