#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set="train_10h"
valid_set="dev"
test_sets="dev_other" #"test_clean test_other dev_clean dev_other"

asr_config=conf/tuning/train_asr_hubert_base_10h_finetuning_espnet.yaml
lm_config=conf/tuning/train_lm_transformer2.yaml
inference_config=conf/decode_asr.yaml

./asr.sh \
    --lang en \
    --ngpu 1 \
    --nj 4 \
    --max_wav_duration 30 \
    --asr_config "${asr_config}" \
    --use_lm false \
    --lm_config "${lm_config}" \
    --inference_config "${inference_config}" \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --bpe_train_text "data/${train_set}/text" \
    --token_type char \
    --lm_train_text "data/${train_set}/text" \
    --asr_stats_dir "asr_stats_raw_en_char_espnet" \
    --feats-normalize null  "$@" 

#data/local/other_text/text
#

# --speed_perturb_factors "0.9 1.0 1.1" \
#    --nbpe 300 \
