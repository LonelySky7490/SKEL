##!/bin/bash

## Config
SEED=54321
MODEL_NAME='fine_tune_baseline_v1_train_fully'
N_EPOCH=5
LR=2e-5
TRAIN_BS=32 
VAL_BS=32 
TEST_BS=32
val_data_type='total' 
test_data_type='total' 
test_strategy_type='v1' 

## Feature Cluster (K=20, P=0.6, Th=1.22) + AMA (Div+Guide)
EXT="AMA_FeatureCluster_K20_Drop0.6_Th1.22" 

## Training
echo ">>> Starting Training with AMA + Feature Clustering..."

# python train.py \
# --lr ${LR} \
# --seed ${SEED} \
# --model ${MODEL_NAME} \
# --snapshot_pref "./ExpResults/${MODEL_NAME}/train_mode_seed${SEED}_bs${TRAIN_BS}_Lr${LR}_evalStrategy_${test_strategy_type}/${EXT}" \
# --n_epoch ${N_EPOCH} \
# --train_batch_size ${TRAIN_BS} \
# --val_batch_size ${VAL_BS} \
# --test_batch_size ${TEST_BS} \
# --val_data_type ${val_data_type} \
# --test_data_type ${test_data_type} \
# --test_strategy_type ${test_strategy_type} \
# --eval_freq 1 \
# --print_iter_freq 100 \
# --clip_gradient 0.8 \
# --num_clusters 10 \
# --center_dropout 0.7 \
# --weight_cl 0.3318 \
# --weight_aux 124.6268 \
# --weight_guide 0.1071 \
# --weight_kd 0.2083 \
# --manifold_token_num 4

## Inference
# echo ">>> Starting Testing..."
# python train.py \
# --lr ${LR} \
# --seed ${SEED} \
# --model ${MODEL_NAME} \
# --evaluate \
# --snapshot_pref "./ExpResults/${MODEL_NAME}/test_mode_seed${SEED}_bs${TRAIN_BS}_Lr${LR}_evalStrategy_${test_strategy_type}/${EXT}" \
# --resume "./ExpResults/task_FullySupervised_best_model.pth.tar" \
# --test_batch_size ${TEST_BS} \
# --test_data_type ${test_data_type} \
# --test_strategy_type ${test_strategy_type} \
# --print_iter_freq 100 \
# --clip_gradient 0.8 \
# --num_clusters 20 \
# --center_dropout 0.0 \
# --dist_threshold 1.22
