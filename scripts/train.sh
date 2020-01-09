set -x
export CUDA_VISIBLE_DEVICES=0

# ============ config ============
export GLUE_DIR=/data/ujval/bert/glue-data
export TASK_NAME=MNLI
MODEL_NAME_OR_PATH=albert-base-v2 # distilbert-base-uncased
OUTPUT_DIR=/data/ujval/results/"$MODEL_NAME_OR_PATH"_"$TASK_NAME"_`date +%Y_%m_%d__%H%M`/
MODEL_TYPE=albert
LOGGING_STEPS=200
SAVE_STEPS=200

# ============ hyperparams ============
BATCH_SIZE=64
MAX_SEQ_LEN=128
LEARNING_RATE=2e-5
NUM_EPOCHS=3.0

python ./examples/run_glue.py \
    --do_train \
    --do_eval \
    --do_lower_case \
    --fp16 \
    --data_dir $GLUE_DIR/$TASK_NAME \
    --model_type $MODEL_TYPE \
    --model_name_or_path $MODEL_NAME_OR_PATH \
    --task_name $TASK_NAME \
    --output_dir $OUTPUT_DIR \
    --num_train_epochs $NUM_EPOCHS \
    --learning_rate $LEARNING_RATE \
    --max_seq_length $MAX_SEQ_LEN \
    --per_gpu_train_batch_size $BATCH_SIZE \
    --evaluate_during_training \
    --logging_steps $LOGGING_STEPS
#    --save_steps $SAVE_STEPS
