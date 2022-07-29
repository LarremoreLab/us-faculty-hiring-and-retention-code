FIJI_USER_DIR=/scratch/Users/kewa8434
FIJI_USFHN_DIR=$FIJI_USER_DIR/faculty-hiring-networks
FIJI_DATASET_DIR=$FIJI_USFHN_DIR/data/datasets/default
FIJI_CONFIGURATION_MODELS_DIR=$FIJI_DATASET_DIR/configuration_models
FIJI_MODEL_STATS_PATH=$FIJI_CONFIGURATION_MODELS_DIR/model_stats.gz

LOCAL_USFHN_DIR=/Users/hne/Documents/research/faculty-hiring-networks
LOCAL_MODEL_STATS_PATH=$LOCAL_USFHN_DIR/data/datasets/default/configuration_models/model_stats.gz

scp fiji:$FIJI_MODEL_STATS_PATH $LOCAL_MODEL_STATS_PATH
