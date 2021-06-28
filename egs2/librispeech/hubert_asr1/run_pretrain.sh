#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

lm_config=conf/tuning/train_lm_transformer2.yaml # didnt' use
inference_config=conf/decode_asr.yaml

# First iteration
fea=mfcc
asr_config=conf/tuning/train_asr_hubert_base_960h_full_pretrain.yaml
#sh ./local/km.sh 500 ${fea}

train_set="train_960_${fea}_km"
valid_set="dev_clean_${fea}_km"
test_sets="test_clean_${fea}_km test_other_${fea}_km dev_clean_${fea}_km dev_other_${fea}_km"

train_set="train_960_km"
valid_set="dev_clean_km"
test_sets="test_clean_km test_other_km dev_clean_km dev_other_km"


./hubert_asr.sh \
    --lang en \
    --ngpu 4 \
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

exit 0
# Secend iteration, extract feature from hubert layer6 and re-generate label
asr_config=conf/tuning/train_asr_hubert_base_960h_full_pretrain_it2.yaml

fea=hubert6

sh ./local/km.sh 500 ${fea}

train_set="train_960_${fea}_km"
valid_set="dev_clean_${fea}_km"
test_sets="test_clean_${fea}_km test_other_${fea}_km dev_clean_${fea}_km dev_other_${fea}_km"

./hubert_asr.sh \
    --lang en \
    --ngpu 4 \
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


