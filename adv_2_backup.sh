#!/bin/bash

K_values=(2)
adv_values=("no" "fgm")
cl_adv_values=("fgsm" "pgd" "FreeAT" "fgm" "no")

for K in "${K_values[@]}"; do
    for adv in "${adv_values[@]}"; do
        for cl_adv in "${cl_adv_values[@]}"; do
          export TEST_FILE=./dataset_processing/MAVEN/valid.jsonl
          export TRAIN_FILE=./dataset_processing/k_shot/fewshot_set/K${K}_MAVEN_0/train.pkl
          export VALID_FILE=./dataset_processing/k_shot/fewshot_set/K${K}_MAVEN_0/dev.pkl
          export LABEL_DICT_PATH=./dataset_processing/k_shot/label_dict_MAVEN.json
          export OUT_PATH=./outputs/MAVEN/${K}-shot/0_${adv}_${cl_adv}
          mkdir -p $OUT_PATH
          echo "Running experiment with K=$K, adv=$adv, cl_adv=$cl_adv"
          echo $OUT_PATH

          python main.py \
              --output_dir $OUT_PATH \
              --train_file $TRAIN_FILE \
              --dev_file $VALID_FILE \
              --test_file $TEST_FILE \
              --label_dict_path $LABEL_DICT_PATH \
              --max_steps 200 \
              --batch_size 4 \
              --eval_batch_size 4 \
              --logging_steps 10 \
              --eval_steps 10 \
              --use_label_semantics \
              --use_normalize \
              --learning_rate 1e-4 \
              --dataset_type MAVEN \
              --queue_size 2048 \
              --start_eval_steps 50 \
              --max_seq_length 192 \
              --fp_16 \
              --drop_none_event \
              --device cuda:1 \
              --wandb fewED_${K}shot_t4_adv \
              --wandbname ${adv}_${cl_adv} \
              --dist_func euclidean \
              --adv $adv \
              --cl_adv $cl_adv
          echo "Experiment completed for K=$K, adv=$adv, cl_adv=$cl_adv"
        done
    done
done