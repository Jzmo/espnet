
#!/usr/bin/env bash

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}
min() {
  local a b
  a=$1
  for b in "$@"; do
      if [ "${b}" -le "${a}" ]; then
          a="${b}"
      fi
  done
  echo "${a}"
}
SECONDS=0

# General configuration
stage=1          # Processes starts from the specified stage.
stop_stage=10000 # Processes is stopped at the specified stage.
skip_data_prep=false # Skip data preparation stages
skip_train=false     # Skip training stages
skip_eval=false      # Skip inference and evaluation stages
skip_upload=true     # Skip packing and uploading stages
ngpu=1           # The number of gpus ("0" uses cpu, otherwise use gpu).
num_nodes=1      # The number of nodes
nj=32            # The number of parallel jobs.
dumpdir=dump     # Directory to dump features.
inference_nj=32     # The number of parallel jobs in inference.
gpu_inference=false # Whether to perform gpu inference.
expdir=exp       # Directory to save experiments.
python=python3       # Specify python to execute espnet commands

# Data preparation related
local_data_opts= # The options given to local/data.sh.

# Speed perturbation related
speed_perturb_factors=  # perturbation factors, e.g. "0.9 1.0 1.1" (separated by space).

# Feature extraction related
feats_type=raw    # Feature type (raw or fbank_pitch).
audio_format=flac # Audio format: wav, flac, wav.ark, flac.ark  (only in feats_type=raw).
fs=16k            # Sampling rate.
min_wav_duration=0.1   # Minimum duration in second
max_wav_duration=20    # Maximum duration in second

# Tokenization related
token_type=bpe      # Tokenization type (char or bpe).
nbpe=30             # The number of BPE vocabulary.
bpemode=unigram     # Mode of BPE (unigram or bpe).
oov="<unk>"         # Out of vocabulary symbol.
blank="<blank>"     # CTC blank symbol
sos_eos="<sos/eos>" # sos and eos symbole
bpe_input_sentence_size=100000000 # Size of input sentence for BPE.
bpe_nlsyms=         # non-linguistic symbols list, separated by a comma, for BPE
bpe_char_cover=1.0  # character coverage when modeling BPE

# Language model related
use_lm=true       # Use language model for ASR decoding.
lm_tag=           # Suffix to the result dir for language model training.
lm_exp=           # Specify the direcotry path for LM experiment.
                  # If this option is specified, lm_tag is ignored.
lm_stats_dir=     # Specify the direcotry path for LM statistics.
lm_config=        # Config for language model training.
lm_args=          # Arguments for language model training, e.g., "--max_epoch 10".
                  # Note that it will overwrite args in lm config.
use_word_lm=false # Whether to use word language model.
num_splits_lm=1   # Number of splitting for lm corpus
# shellcheck disable=SC2034
word_vocab_size=10000 # Size of word vocabulary.
nlsyms_txt=none  # Non-linguistic symbol list if existing.
cleaner=none     # Text cleaner.
score_opts=      # The options given to sclite scoring

# Recognition model related
asr_tag=       # Suffix to the result dir for asr model training.
asr_exp=       # Specify the direcotry path for ASR experiment.
               # If this option is specified, asr_tag is ignored.
asr_stats_dir= # Specify the direcotry path for ASR statistics.
asr_config=    # Config for asr model training.
asr_args=   # Arguments for enhancement model training, e.g., "--max_epoch 10".

# Enhancement model related
enh_exp=    # Specify the direcotry path for enhancement experiment. If this option is specified, enh_tag is ignored.
enh_tag=    # Suffix to the result dir for enhancement model training.
enh_config= # Config for ehancement model training.
enh_args=   # Arguments for enhancement model training, e.g., "--max_epoch 10".
            # Note that it will overwrite args in enhancement config.
spk_num=2   # Number of speakers
noise_type_num=1
dereverb_ref_num=1

# Training data related
use_dereverb_ref=false
use_noise_ref=false

# Pretrained model related
# The number of --init_param must be same.
init_param=

# Enhancement related
inference_tag=    # Suffix to the result dir for decoding.
inference_config= # Config for decoding.
inference_args="--normalize_output_wav true"
inference_model=valid.si_snr.ave.pth

# Evaluation related
scoring_protocol="STOI SDR SAR SIR"
ref_channel=0
score_with_asr=false
asr_exp=""       # asr model for scoring WER
lm_exp=""       # lm model for scoring WER

download_model="kamo-naoyuki/librispeech_asr_train_asr_conformer6_n_fft512_hop_length256_raw_en_bpe5000_scheduler_confwarmup_steps40000_optim_conflr0.0025_sp_valid.acc.ave" # Download a model from Model Zoo and use it for decoding.
# [Task dependent] Set the datadir name created by local/data.sh
train_set=       # Name of training set.
valid_set=       # Name of development set.
test_sets=       # Names of evaluation sets. Multiple items can be specified.
enh_speech_fold_length=800 # fold_length for speech data during enhancement training
bpe_train_text=  # Text file path of bpe training set.
lm_train_text=   # Text file path of language model training set.
lm_dev_text=     # Text file path of language model development set.
lm_test_text=    # Text file path of language model evaluation set.
nlsyms_txt=none  # Non-linguistic symbol list if existing.
cleaner=none     # Text cleaner.
g2p=none         # g2p method (needed if token_type=phn).
lang=noinfo      # The language type of corpus.
score_opts=                # The options given to sclite scoring
local_score_opts=          # The options given to local/score.sh.
asr_speech_fold_length=800 # fold_length for speech data during ASR training.
asr_text_fold_length=150   # fold_length for text data during ASR training.
lm_fold_length=150         # fold_length for LM training.

help_message=$(cat << EOF
Usage: $0 --train-set <train_set_name> --valid-set <valid_set_name> --test_sets <test_set_names>

Options:
    # General configuration
    --stage         # Processes starts from the specified stage (default="${stage}").
    --stop_stage    # Processes is stopped at the specified stage (default="${stop_stage}").
    --skip_data_prep # Skip data preparation stages (default="${skip_data_prep}").
    --skip_train     # Skip training stages (default="${skip_train}").
    --skip_eval      # Skip inference and evaluation stages (default="${skip_eval}").
    --skip_upload    # Skip packing and uploading stages (default="${skip_upload}").
    --ngpu          # The number of gpus ("0" uses cpu, otherwise use gpu, default="${ngpu}").
    --num_nodes     # The number of nodes
    --nj            # The number of parallel jobs (default="${nj}").
    --inference_nj  # The number of parallel jobs in inference (default="${inference_nj}").
    --gpu_inference # Whether to use gpu for inference (default="${gpu_inference}").
    --dumpdir       # Directory to dump features (default="${dumpdir}").
    --expdir        # Directory to save experiments (default="${expdir}").
    --python         # Specify python to execute espnet commands (default="${python}").

    # Data preparation related
    --local_data_opts # The options given to local/data.sh (default="${local_data_opts}").

    # Speed perturbation related
    --speed_perturb_factors   # speed perturbation factors, e.g. "0.9 1.0 1.1" (separated by space, default="${speed_perturb_factors}").

    # Feature extraction related
    --feats_type   # Feature type (only support raw currently).
    --audio_format # Audio format: wav, flac, wav.ark, flac.ark  (only in feats_type=raw, default="${audio_format}").
    --fs           # Sampling rate (default="${fs}").
    --min_wav_duration # Minimum duration in second (default="${min_wav_duration}").
    --max_wav_duration # Maximum duration in second (default="${max_wav_duration}").


    # Enhancemnt model related
    --enh_tag    # Suffix to the result dir for enhancement model training (default="${enh_tag}").
    --enh_config # Config for enhancement model training (default="${enh_config}").
    --enh_args   # Arguments for enhancement model training, e.g., "--max_epoch 10" (default="${enh_args}").
                 # Note that it will overwrite args in enhancement config.
    --spk_num    # Number of speakers in the input audio (default="${spk_num}")
    --noise_type_num   # Number of noise types in the input audio (default="${noise_type_num}")
    --dereverb_ref_num # Number of references for dereverberation (default="${dereverb_ref_num}")

    # Training data related
    --use_dereverb_ref # Whether or not to use dereverberated signal as an additional reference
                         for training a dereverberation model (default="${use_dereverb_ref}")
    --use_noise_ref    # Whether or not to use noise signal as an additional reference
                         for training a denoising model (default="${use_noise_ref}")

    # Pretrained model related
    --init_param    # pretrained model path and module name (default="${init_param}")

    # Enhancement related
    --inference_args   # Arguments for enhancement in the inference stage (default="${inference_args}")
    --inference_model  # Enhancement model path for inference (default="${inference_model}").

    # Evaluation related
    --scoring_protocol    # Metrics to be used for scoring (default="${scoring_protocol}")
    --ref_channel         # Reference channel of the reference speech will be used if the model
                            output is single-channel and reference speech is multi-channel
                            (default="${ref_channel}")

    # [Task dependent] Set the datadir name created by local/data.sh
    --train_set     # Name of training set (required).
    --valid_set       # Name of development set (required).
    --test_sets     # Names of evaluation sets (required).
    --enh_speech_fold_length # fold_length for speech data during enhancement training  (default="${enh_speech_fold_length}").
    --lang         # The language type of corpus (default="${lang}")
EOF
)

log "$0 $*"
# Save command line args for logging (they will be lost after utils/parse_options.sh)
run_args=$(pyscripts/utils/print_args.py $0 "$@")
. utils/parse_options.sh

if [ $# -ne 0 ]; then
    log "${help_message}"
    log "Error: No positional arguments are required."
    exit 2
fi

. ./path.sh
. ./cmd.sh


# Check required arguments
[ -z "${train_set}" ] && { log "${help_message}"; log "Error: --train_set is required"; exit 2; };
[ -z "${valid_set}" ] &&   { log "${help_message}"; log "Error: --valid_set is required"  ; exit 2; };
[ -z "${test_sets}" ] && { log "${help_message}"; log "Error: --test_sets is required"; exit 2; };

data_feats=${dumpdir}/raw


# Use the same text as ASR for bpe training if not specified.
[ -z "${bpe_train_text}" ] && bpe_train_text="${data_feats}/${train_set}/text"
# Use the same text as ASR for lm training if not specified.
[ -z "${lm_train_text}" ] && lm_train_text="${data_feats}/${train_set}/text"
# Use the same text as ASR for lm training if not specified.
[ -z "${lm_dev_text}" ] && lm_dev_text="${data_feats}/${valid_set}/text"
# Use the text of the 1st evaldir if lm_test is not specified
[ -z "${lm_test_text}" ] && lm_test_text="${data_feats}/${test_sets%% *}/text"

# Check tokenization type
if [ "${lang}" != noinfo ]; then
    token_listdir=data/${lang}_token_list
else
    token_listdir=data/token_list
fi
bpedir="${token_listdir}/bpe_${bpemode}${nbpe}"
bpeprefix="${bpedir}"/bpe
bpemodel="${bpeprefix}".model
bpetoken_list="${bpedir}"/tokens.txt
chartoken_list="${token_listdir}"/char/tokens.txt
# NOTE: keep for future development.
# shellcheck disable=SC2034
wordtoken_list="${token_listdir}"/word/tokens.txt
phntoken_list="${token_listdir}"/phn/tokens.txt

if [ "${token_type}" = bpe ]; then
    token_list="${bpetoken_list}"
elif [ "${token_type}" = char ]; then
    token_list="${chartoken_list}"
    bpemodel=none
elif [ "${token_type}" = word ]; then
    token_list="${wordtoken_list}"
    bpemodel=none
elif [ "${token_type}" = phn ]; then
    token_list="${phntoken_list}"
    bpemodel=none
else
    log "Error: not supported --token_type '${token_type}'"
    exit 2
fi
if ${use_word_lm}; then
    log "Error: Word LM is not supported yet"
    exit 2

    lm_token_list="${wordtoken_list}"
    lm_token_type=word
else
    lm_token_list="${token_list}"
    lm_token_type="${token_type}"
fi


# Set tag for naming of model directory
if [ -z "${enh_tag}" ]; then
    if [ -n "${enh_config}" ]; then
        enh_tag="$(basename "${enh_config}" .yaml)_${feats_type}"
    else
        enh_tag="train_${feats_type}"
    fi
    # Add overwritten arg's info
    if [ -n "${enh_args}" ]; then
        enh_tag+="$(echo "${enh_args}" | sed -e "s/--/\_/g" -e "s/[ |=]//g")"
    fi
fi


# The directory used for collect-stats mode
enh_stats_dir="${expdir}/enh_stats_${fs}"
# The directory used for training commands
if [ -z "${enh_exp}" ]; then
enh_exp="${expdir}/enh_${enh_tag}"
fi

if [ -n "${speed_perturb_factors}" ]; then
  enh_stats_dir="${enh_stats_dir}_sp"
  enh_exp="${enh_exp}_sp"
fi

# ========================== Main stages start from here. ==========================

if ! "${skip_data_prep}"; then
    if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
        log "Stage 1: Data preparation for data/${train_set}, data/${valid_set}, etc."
        # [Task dependent] Need to create data.sh for new corpus
        local/data.sh ${local_data_opts}
    fi

    if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
        if ! $use_dereverb_ref && [ -n "${speed_perturb_factors}" ]; then
           log "Stage 2: Speed perturbation: data/${train_set} -> data/${train_set}_sp"

            _scp_list="wav.scp "
            for i in $(seq ${spk_num}); do
                _scp_list+="spk${i}.scp "
            done

           for factor in ${speed_perturb_factors}; do
               if [[ $(bc <<<"${factor} != 1.0") == 1 ]]; then
                   scripts/utils/perturb_enh_data_dir_speed.sh "${factor}" "data/${train_set}" "data/${train_set}_sp${factor}" "${_scp_list}"
                   _dirs+="data/${train_set}_sp${factor} "
               else
                   # If speed factor is 1, same as the original
                   _dirs+="data/${train_set} "
               fi
           done
           utils/combine_data.sh --extra-files "${_scp_list}" "data/${train_set}_sp" ${_dirs}
        else
           log "Skip stage 2: Speed perturbation"
        fi
    fi

    if [ -n "${speed_perturb_factors}" ]; then
        train_set="${train_set}_sp"
    fi

    if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then

        log "Stage 3: Format wav.scp: data/ -> ${data_feats}"

        # ====== Recreating "wav.scp" ======
        # Kaldi-wav.scp, which can describe the file path with unix-pipe, like "cat /some/path |",
        # shouldn't be used in training process.
        # "format_wav_scp.sh" dumps such pipe-style-wav to real audio file
        # and also it can also change the audio-format and sampling rate.
        # If nothing is need, then format_wav_scp.sh does nothing:
        # i.e. the input file format and rate is same as the output.

        for dset in "${train_set}" "${valid_set}" ${test_sets}; do
            if [ "${dset}" = "${train_set}" ] || [ "${dset}" = "${valid_set}" ]; then
                _suf="/org"
            else
                _suf=""
            fi
            utils/copy_data_dir.sh data/"${dset}" "${data_feats}${_suf}/${dset}"
            rm -f ${data_feats}${_suf}/${dset}/{segments,wav.scp,reco2file_and_channel}
            _opts=
            if [ -e data/"${dset}"/segments ]; then
                # "segments" is used for splitting wav files which are written in "wav".scp
                # into utterances. The file format of segments:
                #   <segment_id> <record_id> <start_time> <end_time>
                #   "e.g. call-861225-A-0050-0065 call-861225-A 5.0 6.5"
                # Where the time is written in seconds.
                _opts+="--segments data/${dset}/segments "
            fi


            _spk_list=" "
            for i in $(seq ${spk_num}); do
                _spk_list+="spk${i} "
            done
            if $use_noise_ref && [ -n "${_suf}" ]; then
                # references for denoising ("noise1 noise2 ... niose${noise_type_num} ")
                _spk_list+=$(for n in $(seq $noise_type_num); do echo -n "noise$n "; done)
            fi
            if $use_dereverb_ref && [ -n "${_suf}" ]; then
                # references for dereverberation
                _spk_list+=$(for n in $(seq $dereverb_ref_num); do echo -n "dereverb$n "; done)
            fi

            for spk in ${_spk_list} "wav" ; do
                # shellcheck disable=SC2086
                scripts/audio/format_wav_scp.sh --nj "${nj}" --cmd "${train_cmd}" \
                    --out-filename "${spk}.scp" \
                    --audio-format "${audio_format}" --fs "${fs}" ${_opts} \
                    "data/${dset}/${spk}.scp" "${data_feats}${_suf}/${dset}" \
                    "${data_feats}${_suf}/${dset}/logs/${spk}" "${data_feats}${_suf}/${dset}/data/${spk}"

            done
            echo "${feats_type}" > "${data_feats}${_suf}/${dset}/feats_type"

        done
    fi


    if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
        log "Stage 4: Remove short data: ${data_feats}/org -> ${data_feats}"

        for dset in "${train_set}" "${valid_set}"; do
        # NOTE: Not applying to test_sets to keep original data

            _spk_list=" "
            _scp_list=" "
            for i in $(seq ${spk_num}); do
                _spk_list+="spk${i} "
                _scp_list+="spk${i}.scp "
            done
            if $use_noise_ref; then
                # references for denoising ("noise1 noise2 ... niose${noise_type_num} ")
                _spk_list+=$(for n in $(seq $noise_type_num); do echo -n "noise$n "; done)
                _scp_list+=$(for n in $(seq $noise_type_num); do echo -n "noise$n.scp "; done)                
            fi
            if $use_dereverb_ref; then
                # references for dereverberation
                _spk_list+=$(for n in $(seq $dereverb_ref_num); do echo -n "dereverb$n "; done)
                _scp_list+=$(for n in $(seq $dereverb_ref_num); do echo -n "dereverb$n.scp "; done)
            fi

            # Copy data dir
            utils/copy_data_dir.sh "${data_feats}/org/${dset}" "${data_feats}/${dset}"
            cp "${data_feats}/org/${dset}/feats_type" "${data_feats}/${dset}/feats_type"
            for spk in ${_spk_list};do
                cp "${data_feats}/org/${dset}/${spk}.scp" "${data_feats}/${dset}/${spk}.scp"
            done

            _fs=$(python3 -c "import humanfriendly as h;print(h.parse_size('${fs}'))")
            _min_length=$(python3 -c "print(int(${min_wav_duration} * ${_fs}))")
            _max_length=$(python3 -c "print(int(${max_wav_duration} * ${_fs}))")

            # utt2num_samples is created by format_wav_scp.sh
            <"${data_feats}/org/${dset}/utt2num_samples" \
                awk -v min_length="${_min_length}" -v max_length="${_max_length}" \
                    '{ if ($2 > min_length && $2 < max_length ) print $0; }' \
                    >"${data_feats}/${dset}/utt2num_samples"
            for spk in ${_spk_list} "wav"; do
                <"${data_feats}/org/${dset}/${spk}.scp" \
                    utils/filter_scp.pl "${data_feats}/${dset}/utt2num_samples"  \
                    >"${data_feats}/${dset}/${spk}.scp"
            done

            # fix_data_dir.sh leaves only utts which exist in all files
            utils/fix_data_dir.sh --utt_extra_files "${_scp_list}" "${data_feats}/${dset}"
        done
    fi
else
    log "Skip the data preparation stages"
fi


# ========================== Data preparation is done here. ==========================



if ! "${skip_train}"; then
    if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
        _enh_train_dir="${data_feats}/${train_set}"
        _enh_valid_dir="${data_feats}/${valid_set}"
        log "Stage 5: Enhancement collect stats: train_set=${_enh_train_dir}, valid_set=${_enh_valid_dir}"

        _opts=
        if [ -n "${enh_config}" ]; then
            # To generate the config file: e.g.
            #   % python3 -m espnet2.bin.enh_train --print_config --optim adam
            _opts+="--config ${enh_config} "
        fi

        _scp=wav.scp
        if [[ "${audio_format}" == *ark* ]]; then
            _type=kaldi_ark
        else
            # "sound" supports "wav", "flac", etc.
            _type=sound
        fi

        # 1. Split the key file
        _logdir="${enh_stats_dir}/logdir"
        mkdir -p "${_logdir}"

        # Get the minimum number among ${nj} and the number lines of input files
        _nj=$(min "${nj}" "$(<${_enh_train_dir}/${_scp} wc -l)" "$(<${_enh_valid_dir}/${_scp} wc -l)")

        key_file="${_enh_train_dir}/${_scp}"
        split_scps=""
        for n in $(seq "${_nj}"); do
            split_scps+=" ${_logdir}/train.${n}.scp"
        done
        # shellcheck disable=SC2086
        utils/split_scp.pl "${key_file}" ${split_scps}

        key_file="${_enh_valid_dir}/${_scp}"
        split_scps=""
        for n in $(seq "${_nj}"); do
            split_scps+=" ${_logdir}/valid.${n}.scp"
        done
        # shellcheck disable=SC2086
        utils/split_scp.pl "${key_file}" ${split_scps}

        # 2. Generate run.sh
        log "Generate '${enh_stats_dir}/run.sh'. You can resume the process from stage 5 using this script"
        mkdir -p "${enh_stats_dir}"; echo "${run_args} --stage 5 \"\$@\"; exit \$?" > "${enh_stats_dir}/run.sh"; chmod +x "${enh_stats_dir}/run.sh"

        # 3. Submit jobs
        log "Enhancement collect-stats started... log: '${_logdir}/stats.*.log'"

        # prepare train and valid data parameters
        _train_data_param="--train_data_path_and_name_and_type ${_enh_train_dir}/wav.scp,speech_mix,sound "
        _valid_data_param="--valid_data_path_and_name_and_type ${_enh_valid_dir}/wav.scp,speech_mix,sound "
        for spk in $(seq "${spk_num}"); do
            _train_data_param+="--train_data_path_and_name_and_type ${_enh_train_dir}/spk${spk}.scp,speech_ref${spk},sound "
            _valid_data_param+="--valid_data_path_and_name_and_type ${_enh_valid_dir}/spk${spk}.scp,speech_ref${spk},sound "
        done

        if $use_dereverb_ref; then
            # references for dereverberation
            _train_data_param+=$(for n in $(seq $dereverb_ref_num); do echo -n \
                "--train_data_path_and_name_and_type ${_enh_train_dir}/dereverb${n}.scp,dereverb_ref${n},sound "; done)
            _valid_data_param+=$(for n in $(seq $dereverb_ref_num); do echo -n \
                "--valid_data_path_and_name_and_type ${_enh_valid_dir}/dereverb${n}.scp,dereverb_ref${n},sound "; done)
        fi

        if $use_noise_ref; then
            # references for denoising
            _train_data_param+=$(for n in $(seq $noise_type_num); do echo -n \
                "--train_data_path_and_name_and_type ${_enh_train_dir}/noise${n}.scp,noise_ref${n},sound "; done)
            _valid_data_param+=$(for n in $(seq $noise_type_num); do echo -n \
                "--valid_data_path_and_name_and_type ${_enh_valid_dir}/noise${n}.scp,noise_ref${n},sound "; done)
        fi

        # NOTE: --*_shape_file doesn't require length information if --batch_type=unsorted,
        #       but it's used only for deciding the sample ids.


        # shellcheck disable=SC2086
        ${train_cmd} JOB=1:"${_nj}" "${_logdir}"/stats.JOB.log \
            ${python} -m espnet2.bin.enh_train \
                --collect_stats true \
                --use_preprocessor true \
                ${_train_data_param} \
                ${_valid_data_param} \
                --train_shape_file "${_logdir}/train.JOB.scp" \
                --valid_shape_file "${_logdir}/valid.JOB.scp" \
                --output_dir "${_logdir}/stats.JOB" \
                ${_opts} ${enh_args} || { cat "${_logdir}"/stats.1.log; exit 1; }

        # 4. Aggregate shape files
        _opts=
        for i in $(seq "${_nj}"); do
            _opts+="--input_dir ${_logdir}/stats.${i} "
        done
        # shellcheck disable=SC2086
        ${python} -m espnet2.bin.aggregate_stats_dirs ${_opts} --output_dir "${enh_stats_dir}"

    fi


    if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
        _enh_train_dir="${data_feats}/${train_set}"
        _enh_valid_dir="${data_feats}/${valid_set}"
        log "Stage 6: Enhancemnt Frontend Training: train_set=${_enh_train_dir}, valid_set=${_enh_valid_dir}"

        _opts=
        if [ -n "${enh_config}" ]; then
            # To generate the config file: e.g.
            #   % python3 -m espnet2.bin.enh_train --print_config --optim adam
            _opts+="--config ${enh_config} "
        fi

        _scp=wav.scp
        # "sound" supports "wav", "flac", etc.
        _type=sound
        _fold_length="$((enh_speech_fold_length * 100))"

        # prepare train and valid data parameters
        _train_data_param="--train_data_path_and_name_and_type ${_enh_train_dir}/wav.scp,speech_mix,sound "
        _train_shape_param="--train_shape_file ${enh_stats_dir}/train/speech_mix_shape "
        _valid_data_param="--valid_data_path_and_name_and_type ${_enh_valid_dir}/wav.scp,speech_mix,sound "
        _valid_shape_param="--valid_shape_file ${enh_stats_dir}/valid/speech_mix_shape "
        _fold_length_param="--fold_length ${_fold_length} "
        for spk in $(seq "${spk_num}"); do
            _train_data_param+="--train_data_path_and_name_and_type ${_enh_train_dir}/spk${spk}.scp,speech_ref${spk},sound "
            _train_shape_param+="--train_shape_file ${enh_stats_dir}/train/speech_ref${spk}_shape "
            _valid_data_param+="--valid_data_path_and_name_and_type ${_enh_valid_dir}/spk${spk}.scp,speech_ref${spk},sound "
            _valid_shape_param+="--valid_shape_file ${enh_stats_dir}/valid/speech_ref${spk}_shape "
            _fold_length_param+="--fold_length ${_fold_length} "
        done

        if $use_dereverb_ref; then
            # references for dereverberation
            for n in $(seq "${dereverb_ref_num}"); do
                _train_data_param+="--train_data_path_and_name_and_type ${_enh_train_dir}/dereverb${n}.scp,dereverb_ref${n},sound "
                _train_shape_param+="--train_shape_file ${enh_stats_dir}/train/dereverb_ref${n}_shape "
                _valid_data_param+="--valid_data_path_and_name_and_type ${_enh_valid_dir}/dereverb${n}.scp,dereverb_ref${n},sound "
                _valid_shape_param+="--valid_shape_file ${enh_stats_dir}/valid/dereverb_ref${n}_shape "
                _fold_length_param+="--fold_length ${_fold_length} "
            done
        fi

        if $use_noise_ref; then
            # references for denoising
            for n in $(seq "${noise_type_num}"); do
                _train_data_param+="--train_data_path_and_name_and_type ${_enh_train_dir}/noise${n}.scp,noise_ref${n},sound "
                _train_shape_param+="--train_shape_file ${enh_stats_dir}/train/noise_ref${n}_shape "
                _valid_data_param+="--valid_data_path_and_name_and_type ${_enh_valid_dir}/noise${n}.scp,noise_ref${n},sound "
                _valid_shape_param+="--valid_shape_file ${enh_stats_dir}/valid/noise_ref${n}_shape "
                _fold_length_param+="--fold_length ${_fold_length} "
            done
        fi

        log "Generate '${enh_exp}/run.sh'. You can resume the process from stage 6 using this script"
        mkdir -p "${enh_exp}"; echo "${run_args} --stage 6 \"\$@\"; exit \$?" > "${enh_exp}/run.sh"; chmod +x "${enh_exp}/run.sh"

        log "enh training started... log: '${enh_exp}/train.log'"
        if echo "${cuda_cmd}" | grep -e queue.pl -e queue-freegpu.pl &> /dev/null; then
            # SGE can't include "/" in a job name
            jobname="$(basename ${enh_exp})"
        else
            jobname="${enh_exp}/train.log"
        fi
        # shellcheck disable=SC2086
        ${python} -m espnet2.bin.launch \
            --cmd "${cuda_cmd} --name ${jobname}" \
            --log "${enh_exp}"/train.log \
            --ngpu "${ngpu}" \
            --num_nodes "${num_nodes}" \
            --init_file_prefix "${enh_exp}"/.dist_init_ \
            --multiprocessing_distributed true -- \
            ${python} -m espnet2.bin.enh_train \
                ${_train_data_param} \
                ${_valid_data_param} \
                ${_train_shape_param} \
                ${_valid_shape_param} \
                ${_fold_length_param} \
                --resume true \
                --output_dir "${enh_exp}" \
                ${init_param:+--init_param $init_param} \
                ${_opts} ${enh_args}

    fi
else
    log "Skip the training stages"
fi


if ! "${skip_eval}"; then
    if [ ${stage} -le 7 ] && [ ${stop_stage} -ge 7 ]; then
        log "Stage 7: Enhance Speech: training_dir=${enh_exp}"

        if ${gpu_inference}; then
            _cmd=${cuda_cmd}
            _ngpu=1
        else
            _cmd=${decode_cmd}
            _ngpu=0
        fi

        log "Generate '${enh_exp}/run_enhance.sh'. You can resume the process from stage 7 using this script"
        mkdir -p "${enh_exp}"; echo "${run_args} --stage 7 \"\$@\"; exit \$?" > "${enh_exp}/run_enhance.sh"; chmod +x "${enh_exp}/run_enhance.sh"
        _opts=

        for dset in "${valid_set}" ${test_sets}; do
            _data="${data_feats}/${dset}"
            _dir="${enh_exp}/enhanced_${dset}"
            _logdir="${_dir}/logdir"
            mkdir -p "${_logdir}"

            _scp=wav.scp
            _type=sound

            # 1. Split the key file
            key_file=${_data}/${_scp}
            split_scps=""
            _nj=$(min "${inference_nj}" "$(<${key_file} wc -l)")
            for n in $(seq "${_nj}"); do
                split_scps+=" ${_logdir}/keys.${n}.scp"
            done
            # shellcheck disable=SC2086
            utils/split_scp.pl "${key_file}" ${split_scps}

            # 2. Submit inference jobs
            log "Ehancement started... log: '${_logdir}/enh_inference.*.log'"
            # shellcheck disable=SC2086
            ${_cmd} --gpu "${_ngpu}" JOB=1:"${_nj}" "${_logdir}"/enh_inference.JOB.log \
                ${python} -m espnet2.bin.enh_inference \
                    --ngpu "${_ngpu}" \
                    --fs "${fs}" \
                    --data_path_and_name_and_type "${_data}/${_scp},speech_mix,${_type}" \
                    --key_file "${_logdir}"/keys.JOB.scp \
                    --enh_train_config "${enh_exp}"/config.yaml \
                    --enh_model_file "${enh_exp}"/"${inference_model}" \
                    --output_dir "${_logdir}"/output.JOB \
                    ${_opts} ${inference_args}


            _spk_list=" "
            for i in $(seq ${spk_num}); do
                _spk_list+="spk${i} "
            done

            # 3. Concatenates the output files from each jobs
            for spk in ${_spk_list} ;
            do
                for i in $(seq "${_nj}"); do
                    cat "${_logdir}/output.${i}/${spk}.scp"
                done | LC_ALL=C sort -k1 > "${_dir}/${spk}.scp"
            done

        done
    fi


    if [ ${stage} -le 8 ] && [ ${stop_stage} -ge 8 ]; then
        log "Stage 8: Scoring"
        _cmd=${decode_cmd}

        for dset in "${valid_set}" ${test_sets}; do
            _data="${data_feats}/${dset}"
            _inf_dir="${enh_exp}/enhanced_${dset}"
            _dir="${enh_exp}/enhanced_${dset}/scoring"
            _logdir="${_dir}/logdir"
            mkdir -p "${_logdir}"

            # 1. Split the key file
            key_file=${_data}/wav.scp
            split_scps=""
            _nj=$(min "${inference_nj}" "$(<${key_file} wc -l)")
            for n in $(seq "${_nj}"); do
                split_scps+=" ${_logdir}/keys.${n}.scp"
            done
            # shellcheck disable=SC2086
            utils/split_scp.pl "${key_file}" ${split_scps}


            _ref_scp=
            for spk in $(seq "${spk_num}"); do
                _ref_scp+="--ref_scp ${_data}/spk${spk}.scp "
            done
            _inf_scp=
            for spk in $(seq "${spk_num}"); do
                _inf_scp+="--inf_scp ${_inf_dir}/spk${spk}.scp "
            done

            # 2. Submit scoring jobs
            log "Scoring started... log: '${_logdir}/enh_scoring.*.log'"
            # shellcheck disable=SC2086
            ${_cmd} JOB=1:"${_nj}" "${_logdir}"/enh_scoring.JOB.log \
                ${python} -m espnet2.bin.enh_scoring \
                    --key_file "${_logdir}"/keys.JOB.scp \
                    --output_dir "${_logdir}"/output.JOB \
                    ${_ref_scp} \
                    ${_inf_scp} \
                    --ref_channel ${ref_channel}

            for spk in $(seq "${spk_num}"); do
                for protocol in ${scoring_protocol} wav; do
                    for i in $(seq "${_nj}"); do
                        cat "${_logdir}/output.${i}/${protocol}_spk${spk}"
                    done | LC_ALL=C sort -k1 > "${_dir}/${protocol}_spk${spk}"
                done
            done


            for protocol in ${scoring_protocol}; do
                # shellcheck disable=SC2046
                paste $(for j in $(seq ${spk_num}); do echo "${_dir}"/"${protocol}"_spk"${j}" ; done)  |
                awk 'BEGIN{sum=0}
                    {n=0;score=0;for (i=2; i<=NF; i+=2){n+=1;score+=$i}; sum+=score/n}
                    END{printf ("%.2f\n",sum/NR)}' > "${_dir}/result_${protocol,,}.txt"
            done
        done
        ./scripts/utils/show_enh_score.sh ${enh_exp} > "${enh_exp}/RESULTS.md"

    fi

    asr_exp=${enh_exp}
    if [ -n "${download_model}" ]; then
        log "Use ${download_model} for decoding and evaluation"
        mkdir -p "${asr_exp}/recognition/${download_model}"

        # If the model already exists, you can skip downloading
        espnet_model_zoo_download --unpack true "${download_model}" > "${asr_exp}/recognition/config.txt"

        # Get the path of each file
        _asr_model_file=$(<"${asr_exp}/recognition/config.txt" sed -e "s/.*'asr_model_file': '\([^']*\)'.*$/\1/")
        _asr_train_config=$(<"${asr_exp}/recognition/config.txt" sed -e "s/.*'asr_train_config': '\([^']*\)'.*$/\1/")

        # Create symbolic links
        ln -sf "${_asr_model_file}" "${asr_exp}/recognition"
        ln -sf "${_asr_train_config}" "${asr_exp}/recognition"
        inference_asr_model=$(basename "${_asr_model_file}")

        if [ "$(<${asr_exp}/recognition/config.txt grep -c lm_file)" -gt 0 ]; then
            _lm_file=$(<"${asr_exp}/recognition/config.txt" sed -e "s/.*'lm_file': '\([^']*\)'.*$/\1/")
            _lm_train_config=$(<"${asr_exp}/recognition/config.txt" sed -e "s/.*'lm_train_config': '\([^']*\)'.*$/\1/")

            lm_exp="${asr_exp}/recognition/${download_model}/lm"
            mkdir -p "${lm_exp}"

            ln -sf "${_lm_file}" "${lm_exp}"
            ln -sf "${_lm_train_config}" "${lm_exp}"
            inference_lm=$(basename "${_lm_file}")
        fi

    fi

    inference_args=""

    if [ ${stage} -le 9 ] && [ ${stop_stage} -ge 9 ]; then
        log "Stage 9: Speech Recognition: training_dir=${asr_exp}"

        if ${gpu_inference}; then
            _cmd="${cuda_cmd}"
            _ngpu=1
        else
            _cmd="${decode_cmd}"
            _ngpu=0
        fi

        _opts=
        if [ -n "${inference_config}" ]; then
            _opts+="--config ${inference_config} "
        fi
        if "${use_lm}"; then
            if "${use_word_lm}"; then
                _opts+="--word_lm_train_config ${lm_exp}/config.yaml "
                _opts+="--word_lm_file ${lm_exp}/${inference_lm} "
            else
                _opts+="--lm_train_config ${lm_exp}/config.yaml "
                _opts+="--lm_file ${lm_exp}/${inference_lm} "
            fi
        fi

        # 2. Generate run.sh
        log "Generate '${asr_exp}/${inference_tag}/run.sh'. You can resume the process from stage 13 using this script"
        mkdir -p "${asr_exp}/${inference_tag}"; echo "${run_args} --stage 13 \"\$@\"; exit \$?" > "${asr_exp}/${inference_tag}/run.sh"; chmod +x "${asr_exp}/${inference_tag}/run.sh"

        for dset in ${valid_set} ${test_sets}; do
        # for dset in ${valid_set}; do
            _data="${data_feats}/${dset}"
            _dir="${asr_exp}/recognition/decode_${dset}"

            _enh_inf_dir="${asr_exp}/enhanced_${dset}"
            for spk in $(seq "${spk_num}"); do
                scripts/audio/format_wav_scp.sh --nj "${nj}" --cmd "${train_cmd}" \
                        --out-filename "resampled_spk${spk}.scp" \
                        --audio-format "${audio_format}" --fs "16000" \
                        "${_enh_inf_dir}/spk${spk}.scp" "${_dir}/resampled_audios" \
                        "${_dir}/resampled_audios/logs/spk${spk}" "${_dir}/resampled_audios/data/spk${spk}"

                _scp=resampled_spk${spk}.scp
                _type=sound

                _logdir="${_dir}/logdir_spk${spk}"
                mkdir -p "${_logdir}"

                # 1. Split the key file
                key_file=${_dir}/resampled_audios/${_scp}
                split_scps=""
                _nj=$(min "${inference_nj}" "$(<${key_file} wc -l)")
                for n in $(seq "${_nj}"); do
                    split_scps+=" ${_logdir}/keys.${n}.scp"
                done
                # shellcheck disable=SC2086
                utils/split_scp.pl "${key_file}" ${split_scps}

                # 2. Submit decoding jobs
                log "Decoding started... log: '${_logdir}/asr_inference.*.log'"
                # shellcheck disable=SC2086
                ${_cmd} --gpu "${_ngpu}" JOB=1:"${_nj}" "${_logdir}"/asr_inference.JOB.log \
                    ${python} -m espnet2.bin.asr_inference \
                        --ngpu "${_ngpu}" \
                        --data_path_and_name_and_type "${_dir}/resampled_audios/${_scp},speech,${_type}" \
                        --key_file "${_logdir}"/keys.JOB.scp \
                        --asr_train_config "${asr_exp}"/recognition/config.yaml \
                        --asr_model_file "${asr_exp}"/recognition/"${inference_asr_model}" \
                        --output_dir "${_logdir}"/output.JOB \
                        ${_opts} ${inference_args}

                # 3. Concatenates the output files from each jobs
                for f in token token_int score text; do
                    for i in $(seq "${_nj}"); do
                        cat "${_logdir}/output.${i}/1best_recog/${f}"
                    done | LC_ALL=C sort -k1 >"${_dir}/${f}_spk${spk}"
                done
            done
        done
    fi


    if [ ${stage} -le 10 ] && [ ${stop_stage} -ge 10 ]; then
        log "Stage 10: ASR Scoring"
        token_type="word"
        if [ "${token_type}" = phn ]; then
            log "Error: Not implemented for token_type=phn"
            exit 1
        fi

        for dset in ${valid_set} ${test_sets}; do
        # for dset in ${test_sets}; do
            _data="${data_feats}/${dset}"
            _dir="${asr_exp}/recognition/decode_${dset}"

            for _type in cer wer ter; do
                [ "${_type}" = ter ] && [ ! -f "${bpemodel}" ] && continue

                _scoredir="${_dir}/score_${_type}"
                mkdir -p "${_scoredir}"

                if [ "${_type}" = wer ]; then
                    token_type="word"
                    opts=""
                elif [ "${_type}" = cer ]; then
                    token_type="char"
                    opts=""
                elif [ "${_type}" = ter ]; then
                    token_type="bpe"
                    opts="--bpemodel ${bpemodel} "
                fi

                for spk in $(seq "${spk_num}"); do
                    # Tokenize text to word level
                    paste \
                        <(<"${_data}/text_spk${spk}" \
                            ${python} -m espnet2.bin.tokenize_text  \
                                -f 2- --input - --output - \
                                --token_type ${token_type} \
                                --non_linguistic_symbols "${nlsyms_txt}" \
                                --remove_non_linguistic_symbols true \
                                --cleaner "${cleaner}" \
                                ${opts} \
                                ) \
                        <(<"${_data}/text_spk${spk}" awk '{ print "(" $1 ")" }') \
                            >"${_scoredir}/ref_spk${spk}.trn"

                    # NOTE(kamo): Don't use cleaner for hyp
                    paste \
                        <(<"${_dir}/text_spk${spk}"  \
                            ${python} -m espnet2.bin.tokenize_text  \
                                -f 2- --input - --output - \
                                --token_type ${token_type} \
                                --non_linguistic_symbols "${nlsyms_txt}" \
                                --remove_non_linguistic_symbols true \
                                ${opts} \
                                ) \
                        <(<"${_data}/text_spk${spk}" awk '{ print "(" $1 ")" }') \
                            >"${_scoredir}/hyp_spk${spk}.trn"
                done

                # PIT scoring
                for r_spk in $(seq "${spk_num}"); do
                    for h_spk in $(seq "${spk_num}"); do
                        sclite \
                            -r "${_scoredir}/ref_spk${r_spk}.trn" trn \
                            -h "${_scoredir}/hyp_spk${h_spk}.trn" trn \
                            -i rm -o all stdout > "${_scoredir}/result_r${r_spk}h${h_spk}.txt"
                    done
                done

                scripts/utils/eval_perm_free_error.py --num-spkrs ${spk_num} \
                    --results-dir ${_scoredir}

                sclite \
                    ${score_opts} \
                    -r "${_scoredir}/ref.trn" trn \
                    -h "${_scoredir}/hyp.trn" trn \
                    -i rm -o all stdout > "${_scoredir}/result.txt"

                log "Write ${_type} result in ${_scoredir}/result.txt"
                grep -e Avg -e SPKR -m 2 "${_scoredir}/result.txt"
            done
        done

        [ -f local/score.sh ] && local/score.sh ${local_score_opts} "${asr_exp}"

        # Show results in Markdown syntax
        scripts/utils/show_asr_result.sh "${asr_exp}" > "${asr_exp}"/RESULTS_ASR.md
        cat "${asr_exp}"/RESULTS_ASR.md

    fi
else
    log "Skip the evaluation stages"
fi

log "Successfully finished. [elapsed=${SECONDS}s]"