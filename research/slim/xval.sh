#!/bin/bash
DATASET_DIR=/workspace/data/Part-A_Originaljpeg/
TRAIN_DIR=/workspace/results/Xval2/train_logs/
EVAL_DIR=/workspace/results/Xval2/eval_logs/

for i in `seq 0 9`;
do
	echo "**************************************Iteration $i******************************************"
	TRAIN_DIR=/workspace/results/Xval2/train_logs/val_${i}/
	echo "Making $TRAIN_DIR"
	mkdir $TRAIN_DIR
        
        echo "********************************Training $i***************************************************"
        CHECKPOINT_PATH=/workspace/data/checkpoints/resnet_v1_50.ckpt
	SPLIT=train_$i
        CUDA_VISIBLE_DEVICES=0 python train_image_classifier.py \
        --train_dir=$TRAIN_DIR --dataset_dir=$DATASET_DIR --dataset_name=pathology \
        --dataset_split_name=$SPLIT --model_name=resnet_v1_50_pathology_benchmark \
        --checkpoint_path=$CHECKPOINT_PATH \
        --batch_size 2 --learning_rate=0.000625 --end_learning_rate=0.00000625 --num_clones=1 \
        --checkpoint_exclude_scopes=resnet_v1_50_pathology_benchmark/logits,resnet_v1_50_pathology_benchmark/fc1 \
        --trainable_scopes=resnet_v1_50_pathology_benchmark/logits,resnet_v1_50_pathology_benchmark/fc1 \
        --map_checkpoint=True \
        --max_number_of_steps 8000

        echo "***********************************Evaluating $i************************************"
        CHECKPOINT_PATH=${TRAIN_DIR}model.ckpt-8000
	SPLIT=validation_$i
        CUDA_VISIBLE_DEVICES=0 python eval_image_classifier.py --alsologtostderr --batch_size=2 \
        --checkpoint_path=$CHECKPOINT_PATH --dataset_dir=$DATASET_DIR \
        --eval_dir=$EVAL_DIR \
        --dataset_name=pathology --dataset_split_name=$SPLIT \
        --model_name=resnet_v1_50_pathology_benchmark
done

