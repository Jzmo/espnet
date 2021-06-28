#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set="train_960_km"
valid_set="dev_clean_km"
test_sets="test_clean_km test_other_km dev_clean_km dev_other_km"

asr_config=conf/tuning/train_asr_hubert_base_960h_full_pretrain_gpu32.yaml
lm_config=conf/tuning/train_lm_transformer2.yaml
inference_config=conf/decode_asr.yaml


#sh ./local/km.sh 500

./hubert_asr.sh \
    --lang en \
    --ngpu 8 \
    --num_nodes 4 \
    --nj 32 \
    --max_wav_duration 30 \
    --asr_config "${asr_config}" \
    --use_lm false \
    --lm_config "${lm_config}" \
    --inference_config "${inference_config}" \
    --train-set "${train_set}" \
    --valid-set "${valid_set}" \
    --test_sets "${test_sets}" \
    --lm_train_text "data/${train_set}/text" \
    --feats-normalize null \
    --token_type word "$@"
