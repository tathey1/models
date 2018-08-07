#!/bin/bash

DATASET_DIR=/workspace/data/Part-A_Originaljpeg/
TRAIN_DIR=/workspace/results/MCXval/checkpoints/
EVAL_DIR=/workspace/results/MCXval/eval_logs/

for i in `seq 1 1`;
do
	echo "Removing files from $TRAIN_DIR"
	rm ${TRAIN_DIR}*
	echo "Removing tfrecords and txts from $DATASET_DIR"
	rm ${DATASET_DIR}*.tfrecord
	rm ${DATASET_DIR}*.txt

	echo "**************************************Iteration $i******************************************"
	echo "********************************Making split $i************************************************"
	python download_and_convert_data.py --dataset_name=pathology \
	--dataset_dir=$DATASET_DIR

	echo "********************************Training $i***************************************************"
	CHECKPOINT_PATH=/workspace/data/checkpoints/resnet_v1_50.ckpt
	CUDA_VISIBLE_DEVICES=0,1,2 python train_image_classifier.py \
	--train_dir=$TRAIN_DIR --dataset_dir=$DATASET_DIR --dataset_name=pathology \
	--dataset_split=train --model_name=resnet_v1_50_pathology_benchmark \
	--checkpoint_path=$CHECKPOINT_PATH \
	--batch_size 2 --learning_rate=0.000625 --end_learning_rate=0.00000625 \
	--checkpoint_exclude_scopes=resnet_v1_50_pathology_benchmark/logits,resnet_v1_50_pathology_benchmark/fc1 \
	--trainable_scopes=resnet_v1_50_pathology_benchmark/logits,resnet_v1_50_pathology_benchmark/fc1 \
	--map_checkpoint=True \
	--max_number_of_steps 1
	
	echo "***********************************Evaluating $i************************************"
	CHECKPOINT_PATH=${TRAIN_DIR}model.ckpt-1
	CUDA_VISIBLE_DEVICES=0 python eval_image_classifier.py --alsologtostderr --batch_size=2 \
	--checkpoint_path=$CHECKPOINT_PATH --dataset_dir=$DATASET_DIR \
	--eval_dir=$EVAL_DIR \
	--dataset_name=pathology --dataset_split_name=validation \
	--model_name=resnet_v1_50_pathology_benchmark
done
