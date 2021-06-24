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
stop_stage=10
train_set="train_960"
dev_set="dev"
test_set="dev_clean dev_other test_clean test_other"

log "$0 $*"
. utils/parse_options.sh

. ./db.sh
. ./path.sh
. ./cmd.sh

if [ $# -ne 1 ]; then
    echo "Usage: $0 <n-cluster>"
    echo "e.g.: $0 50"
    exit 0
fi

n_cluster=$1

python=python3       # Specify python to execute espnet commands.
nj=32

# generate data/train_set/...

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    log "Stage 1: Data preparation for data/${train_set}, data/${test_set}, etc."
    # [Task dependent] Need to create data.sh for new corpus
    local/data.sh 
fi

# k-means clustering
# mfcc or hubert
# if mfcc: check if mfcc exist, if not generate
# if hubert: check if hubert feature exit, if not, check if hubert model exist
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    log "stage 2: Learn K-means with MFCC feature based on scikit-learn"
    _km_dir=exp/km/${train_set}/results/
    mkdir -p ${_km_dir}
    ${python} local/sklearn_km.py \
              --root data/${train_set} \
	      --km-path ${_km_dir} \
	      --n-cluster ${n_cluster} \
	      --portion 0.1 \
	      --nj ${nj}
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    log "stage 4: Generate K-means pseudo label"
    _km_dir=exp/km/${train_set}/results/
    mkdir -p ${_km_dir}
    for task in ${dev_set} ${train_set} ${test_set}; do
	${python} local/dump_km_label.py \
		  --km-path ${_km_dir} \
		  --recog-set data/${task} \
		  --nj ${nj}

	# move data/${task}/ to new folders and rename ptext
	if [[ -d "data/${task}_km" ]]; then
	    echo "data/${task}_km already exists, will remove it"
	    rm -r data/${task}_km
	fi
	cp -r data/${task} data/${task}_km
	cat data/${task}_km/ptext| sort > data/${task}_km/text
	utils/validate_data_dir.sh --no-feats data/${task}_km || exit 1
	rm data/${task}_km/ptext
    done
fi


if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    log "stage 5: Generate char-based fairseq style dictionary: <token> <count>"
    # generate dictionaries
    cat data/${train_set}_km/text | cut -d" " -f2- > data/${train_set}_km/train.txt
    ${python} -m espnet2.bin.tokenize_text  \
              --token_type char \
              --input "data/{train_set}_km/train.txt" --output "data/{train_set}_km/dict.txt" ${_opts} \
              --field 2- \
              --cleaner none \
              --g2p none \
              --write_vocabulary true \
	      --write-word-and-count true \
              --add_symbol "${blank}:0" \
              --add_symbol "${oov}:1" \
              --add_symbol "${sos_eos}:-1"
	if [[ -e "data/{train_set}_km/dict.txt" ]]; then
	    echo "Successfully generate the data/{train_set}_km/dict.txt"
	fi
fi

