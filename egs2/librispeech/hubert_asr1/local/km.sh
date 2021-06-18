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

stage=1
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

data_dir=${LIBRISPEECH}/LibriSpeech/
python=python3       # Specify python to execute espnet commands.
nj=32


# k-means clustering
# mfcc or hubert
# if mfcc: check if mfcc exist, if not generate
# if hubert: check if hubert feature exit, if not, check if hubert model exist
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    log "stage 2: "
    _km_dir=exp/km/${train_set}/results/
    mkdir -p ${_km_dir}
    ${python} local/sklearn_km.py \
              --root data/${train_set} \
	      --km-path ${_km_dir} \
	      --n-cluster ${n_cluster} \
	      --nj ${nj}
fi


if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    log "stage 3: "
    _km_dir=exp/km/${train_set}/results/
    mkdir -p ${_km_dir}
    for task in ${dev_set} ${train_set} ${test_set}; do
	${python} local/dump_km_label.py \
		  --km-path ${_km_dir} \
		  --recog-set data/${task} \
		  --nj ${nj}
    done
fi


if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    log "stage 4: "
    for task in ${dev_set} ${train_set} ${test_set}; do
	# move data/${task}/ to new folders and rename ptext
	if [[ -d "data/${task}_km" ]]; then
	    echo "data/${task}_km already exists, will remove it"
	    rm -r data/${task}_km
	fi
	cp -r data/${task} data/${task}_km
	mv data/${task}_km/ptext data/${task}_km/text
	utils/validate_data_dir.sh --no-feats data/${task}_km || exit 1
    done
fi
if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
    log "stage 5: "
    # generate dictionaries
    cat data/${train_set}_km/text | cut -d" " -f2- > data/${train_set}_km/train.txt
    if [[ -d "data-bin" ]]; then
	rm -r data-bin
    fi    
    python -m fairseq_cli.preprocess \
	   --dataset-impl mmap \
	   --trainpref  data/${train_set}_km/train.txt\
	   --only-source  \
	   --thresholdsrc 0
	if [[ -e "data-bin/dict.txt" ]]; then
	    echo "Successfully generate the data-bin/dict.txt"
	fi
fi

