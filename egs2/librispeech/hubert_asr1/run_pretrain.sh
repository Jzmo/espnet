#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set="train_clean_100_km"
valid_set="dev_km"
test_sets="test_clean_km test_other_km dev_clean_km dev_other_km"

asr_config=conf/tuning/train_asr_hubert_base_960h_pretrain.yaml
lm_config=conf/tuning/train_lm_transformer2.yaml
inference_config=conf/decode_asr.yaml

./hubert_asr.sh \
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
    --lm_train_text "data/${train_set}/text" \
    --token_type word "$@"

#    --nbpe 5000 \
#--bpe_train_text "data/${train_set}/text" "$@"
# --speed_perturb_factors "0.9 1.0 1.1" \
