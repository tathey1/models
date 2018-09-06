: '
Performs 10 fold cross validation

in this form, there must be tf.records for the different fold splits
i.e. 20 files: pathology_training_0.tfrecord, pathology_training_1.tfrecord,...,pathology_validation_0.tfrecord, pathology_validation_1.tfrecord

Things that I often modify:
NUM_STEPS, DATASET_DIR, EVAL_DIR, TRAIN_DIR
model_name argument in both training and evaluating
batch_size argument in both training and evaluating (e.g. tiling is more computationally intensize and therefore can only handle smaller batch_sizes)
learning_rate and end_learning_rate, I usually scale them with the batch size (bigger batch_size, bigger learning rate)
trainable_scopes, and checkpoint_exclude_scopes are usually modified when you switch models
map_checkpoint is true when you are borrowing pretrained weights but using them in a model of another name
'

#!/bin/bash
NUM_STEPS=15000
DATASET_DIR=/workspace/data/Part-A_Originaljpeg/
EVAL_DIR=/workspace/results/xval_final_15000/eval_logs/

for i in `seq 0 9`;
do
	echo "**************************************Iteration $i******************************************"
	TRAIN_DIR=/workspace/results/xval_final_15000/train_logs/val_${i}/
	
	echo "Making $TRAIN_DIR"
	mkdir $TRAIN_DIR
        
        echo "********************************Training $i***************************************************"
        CHECKPOINT_PATH=/workspace/data/checkpoints/resnet_v1_50.ckpt
	SPLIT=train_$i
        CUDA_VISIBLE_DEVICES=0 python train_image_classifier.py \
        --train_dir=$TRAIN_DIR --dataset_dir=$DATASET_DIR --dataset_name=pathology \
        --dataset_split_name=$SPLIT --model_name=resnet_v1_50_final \
        --checkpoint_path=$CHECKPOINT_PATH \
        --batch_size=2 --learning_rate=0.000625 --end_learning_rate=0.00000625 --num_clones=1 \
        --checkpoint_exclude_scopes=resnet_v1_50_final/logits \
        --trainable_scopes=resnet_v1_50_final/logits \
        --map_checkpoint=True \
        --max_number_of_steps $NUM_STEPS
	
        echo "***********************************Evaluating $i************************************"
        CHECKPOINT_PATH=${TRAIN_DIR}model.ckpt-$NUM_STEPS
	SPLIT=validation_$i
        CUDA_VISIBLE_DEVICES=0 python eval_image_classifier.py --alsologtostderr --batch_size=2 \
        --checkpoint_path=$CHECKPOINT_PATH --dataset_dir=$DATASET_DIR \
        --eval_dir=$EVAL_DIR \
        --dataset_name=pathology --dataset_split_name=$SPLIT \
        --model_name=resnet_v1_50_final
done

