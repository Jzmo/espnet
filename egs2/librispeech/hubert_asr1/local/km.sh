#!/bin/bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

stage=3
stop_stage=4
train_set="train_960"
#train_set="train_clean_100"
dev_set="dev"
test_set="dev_clean dev_other test_clean test_other"

#hubert_url="espnet"
#hubert_dir_path="exp/asr_train_asr_hubert_base_960h_full_pretrain_raw_en_word/"
hubert_url="https://dl.fbaipublicfiles.com/hubert/hubert_base_ls960.pt"
hubert_dir_path="./downloads/hubert_pretrained_models/hubert_base_ls960.pt"
portion=0.1

log "$0 $*"

. ./db.sh
. ./path.sh
. ./cmd.sh

if [ $# -ne 2 ]; then
    echo "Usage: $0 <n-cluster:500> <feature:mfcc>"
    echo "e.g.: $0 50"
    exit 0
fi

n_cluster=$1
feature=$2 # mfcc or hubert#layer

python=python3       # Specify python to execute espnet commands.
nj=32

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    log "Stage 1: Data preparation for data/${train_set}, data/${test_set}, etc."
    # [Task dependent] Need to create data.sh for new corpus
    local/data.sh 
fi

# k-means clustering
# mfcc or hubert based
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    log "stage 2: Learn K-means with ${feature} feature based on scikit-learn"
    _km_dir=exp/km/${train_set}_${feature}/results/
    mkdir -p ${_km_dir}
    ${python} local/sklearn_km.py \
              --root data/${train_set} \
	      --km-path ${_km_dir} \
	      --n-cluster ${n_cluster} \
	      --portion "${portion}" \
	      --feature "${feature}" \
	      --hurl "${hubert_url}" \
	      --hdir "${hubert_dir_path}" \
	      --nj ${nj}
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    log "stage 3: Generate K-means pseudo label"
    km_dir=exp/km/${train_set}_${feature}/results/
    mkdir -p ${km_dir}
    for task in ${dev_set} ${train_set} ${test_set}; do
	${python} local/dump_km_label.py \
		  --km-path ${km_dir} \
		  --feature "${feature}" \
		  --hurl "${hubert_url}" \
		  --hdir "${hubert_dir_path}" \
		  --recog-set data/${task} \
		  --nj ${nj}

	# move data/${task}/ to new folders and rename ptext
	plabel_dir="data/${task}_${feature}_km"
	if [[ -d "${plabel_dir}" ]]; then
	    echo "${plabel_dir} already exists, will remove it"
	    rm -r ${plabel_dir}
	fi
	cp -r data/${task} ${plabel_dir}
	cat ${plabel_dir}/ptext| sort > ${plabel_dir}/text
	utils/validate_data_dir.sh --no-feats ${plabel_dir} || exit 1
	rm ${plabel_dir}/ptext
    done
fi


if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    log "stage 4: Generate char-based fairseq style dictionary: <token> <count>"
    # generate dictionaries
    oov="<unk>"         # Out of vocabulary symbol.
    blank="<blank>"     # CTC blank symbol
    sos_eos="<sos/eos>" # sos and eos symbole

    cat data/${train_set}_${feature}_km/text | cut -d" " -f2- > data/${train_set}_${feature}_km/train.txt
    ${python} -m espnet2.bin.tokenize_text  \
              --token_type char \
              --input "data/${train_set}_${feature}_km/train.txt" --output "data/${train_set}_${feature}_km/dict.txt" \
	      --non_linguistic_symbols none \
              --field 2- \
              --cleaner none \
              --g2p none \
              --write_vocabulary true \
	      --write-word-and-count true \
              --add_symbol "${blank}:0" \
              --add_symbol "${oov}:1" \
              --add_symbol "${sos_eos}:-1"
	if [[ -e "data/{train_set}_${feature}_km/dict.txt" ]]; then
	    echo "Successfully generate the data/{train_set}_${feature}_km/dict.txt"
	fi
fi

