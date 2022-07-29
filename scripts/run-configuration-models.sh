FIJI_USER_DIR=/scratch/Users/kewa8434
FIJI_USFHN_DIR=$FIJI_USER_DIR/faculty-hiring-networks
FIJI_DATASET_DIR=$FIJI_USFHN_DIR/data/datasets/default
FIJI_CONFIGURATION_MODELS_DIR=$FIJI_DATASET_DIR/configuration_models

FIJI_DATA_PATH=$FIJI_DATASET_DIR/data.gz
FIJI_HIERARCHY_STATS_PATH=$FIJI_USFHN_DIR/data/stats/dataframes/ranks/prestige/hierarchy-stats.gz

FIJI_JOBS_DIR=$FIJI_USFHN_DIR/jobs/configuration-models

FIJI_ERR_DIR=$FIJI_USER_DIR/err
FIJI_LOG_DIR=$FIJI_USER_DIR/log

LOCAL_USFHN_DIR=/Users/hne/Documents/research/faculty-hiring-networks
LOCAL_DATA_PATH=$LOCAL_USFHN_DIR/data/datasets/default/data.gz
LOCAL_HIERARCHY_STATS_PATH=$LOCAL_USFHN_DIR/data/stats/dataframes/ranks/prestige/hierarchy-stats.gz

# remove outdated data and create required directories
ssh fiji rm -fr $FIJI_DATASET_DIR
ssh fiji mkdir $FIJI_DATASET_DIR
ssh fiji mkdir $FIJI_CONFIGURATION_MODELS_DIR

# copy required data over
scp $LOCAL_DATA_PATH fiji:$FIJI_DATA_PATH
scp $LOCAL_HIERARCHY_STATS_PATH fiji:$FIJI_HIERARCHY_STATS_PATH

# clean the log and error directories
ssh fiji rm -fr $FIJI_ERR_DIR
ssh fiji mkdir $FIJI_ERR_DIR
ssh fiji rm -fr $FIJI_LOG_DIR
ssh fiji mkdir $FIJI_LOG_DIR

# submit jobs to rerun the configuration models
ssh fiji sbatch $FIJI_JOBS_DIR/academia.sbatch
ssh fiji sbatch $FIJI_JOBS_DIR/umbrellas.sbatch
ssh fiji sbatch $FIJI_JOBS_DIR/fields.sbatch
sleep 5
ssh fiji sbatch $FIJI_JOBS_DIR/academia.sbatch
ssh fiji sbatch $FIJI_JOBS_DIR/umbrellas.sbatch
ssh fiji sbatch $FIJI_JOBS_DIR/fields.sbatch
