#!/bin/bash

for K in 5
do
    for idx in {0..9}
    do
        export TEST_FILE=./dataset_processing/MAVEN/valid.jsonl
        export TRAIN_FILE=./dataset_processing/k_shot/fewshot_set/K${K}_MAVEN_${idx}/train.pkl
        export VALID_FILE=./dataset_processing/k_shot/fewshot_set/K${K}_MAVEN_${idx}/dev.pkl
        export LABEL_DICT_PATH=./dataset_processing/k_shot/label_dict_MAVEN.json
        export OUT_PATH=./outputs/MAVEN/${K}-shot/${idx}
        mkdir -p $OUT_PATH
        echo "Processing K=${K}, idx=${idx}, Output Path: $OUT_PATH"

        python main.py \
            --output_dir $OUT_PATH \
            --train_file $TRAIN_FILE \
            --dev_file $VALID_FILE \
            --test_file $TEST_FILE \
            --label_dict_path $LABEL_DICT_PATH \
            --max_steps 200 \
            --batch_size 32 \
            --logging_steps 10 \
            --eval_steps 10 \
            --use_label_semantics \
            --use_normalize \
            --learning_rate 1e-4 \
            --dataset_type MAVEN \
            --queue_size 8192 \
            --start_eval_steps 50 \
            --max_seq_length 192 \
            --fp_16 \
            --drop_none_event

        if [ $? -ne 0 ]; then
            echo "Error occurred with K=${K}, idx=${idx}. Continuing to next iteration."
            continue
        fi
    done
done

