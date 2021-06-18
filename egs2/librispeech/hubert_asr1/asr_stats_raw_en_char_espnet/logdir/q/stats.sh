#!/bin/bash
cd /ocean/projects/cis210027p/jzmo/work/espnet/egs2/librispeech/hubert_asr1
. ./path.sh
( echo '#' Running on `hostname`
  echo '#' Started at `date`
  set | grep SLURM | while read line; do echo "# $line"; done
  echo -n '# '; cat <<EOF
python3 -m espnet2.bin.asr_train --collect_stats true --use_preprocessor true --bpemodel none --token_type char --token_list data/en_token_list/char/tokens.txt --non_linguistic_symbols none --cleaner none --g2p none --train_data_path_and_name_and_type dump/raw/train_10h/wav.scp,speech,sound --train_data_path_and_name_and_type dump/raw/train_10h/text,text,text --valid_data_path_and_name_and_type dump/raw/dev/wav.scp,speech,sound --valid_data_path_and_name_and_type dump/raw/dev/text,text,text --train_shape_file asr_stats_raw_en_char_espnet/logdir/train.${SLURM_ARRAY_TASK_ID}.scp --valid_shape_file asr_stats_raw_en_char_espnet/logdir/valid.${SLURM_ARRAY_TASK_ID}.scp --output_dir asr_stats_raw_en_char_espnet/logdir/stats.${SLURM_ARRAY_TASK_ID} --config conf/tuning/train_asr_hubert_base_10h_finetuning_espnet.yaml --frontend_conf fs=16k 
EOF
) >asr_stats_raw_en_char_espnet/logdir/stats.$SLURM_ARRAY_TASK_ID.log
if [ "$CUDA_VISIBLE_DEVICES" == "NoDevFiles" ]; then
  ( echo CUDA_VISIBLE_DEVICES set to NoDevFiles, unsetting it... 
  )>>asr_stats_raw_en_char_espnet/logdir/stats.$SLURM_ARRAY_TASK_ID.log
  unset CUDA_VISIBLE_DEVICES
fi
time1=`date +"%s"`
 ( python3 -m espnet2.bin.asr_train --collect_stats true --use_preprocessor true --bpemodel none --token_type char --token_list data/en_token_list/char/tokens.txt --non_linguistic_symbols none --cleaner none --g2p none --train_data_path_and_name_and_type dump/raw/train_10h/wav.scp,speech,sound --train_data_path_and_name_and_type dump/raw/train_10h/text,text,text --valid_data_path_and_name_and_type dump/raw/dev/wav.scp,speech,sound --valid_data_path_and_name_and_type dump/raw/dev/text,text,text --train_shape_file asr_stats_raw_en_char_espnet/logdir/train.${SLURM_ARRAY_TASK_ID}.scp --valid_shape_file asr_stats_raw_en_char_espnet/logdir/valid.${SLURM_ARRAY_TASK_ID}.scp --output_dir asr_stats_raw_en_char_espnet/logdir/stats.${SLURM_ARRAY_TASK_ID} --config conf/tuning/train_asr_hubert_base_10h_finetuning_espnet.yaml --frontend_conf fs=16k  ) &>>asr_stats_raw_en_char_espnet/logdir/stats.$SLURM_ARRAY_TASK_ID.log
ret=$?
sync || true
time2=`date +"%s"`
echo '#' Accounting: begin_time=$time1 >>asr_stats_raw_en_char_espnet/logdir/stats.$SLURM_ARRAY_TASK_ID.log
echo '#' Accounting: end_time=$time2 >>asr_stats_raw_en_char_espnet/logdir/stats.$SLURM_ARRAY_TASK_ID.log
echo '#' Accounting: time=$(($time2-$time1)) threads=1 >>asr_stats_raw_en_char_espnet/logdir/stats.$SLURM_ARRAY_TASK_ID.log
echo '#' Finished at `date` with status $ret >>asr_stats_raw_en_char_espnet/logdir/stats.$SLURM_ARRAY_TASK_ID.log
[ $ret -eq 137 ] && exit 100;
touch asr_stats_raw_en_char_espnet/logdir/q/done.3765050.$SLURM_ARRAY_TASK_ID
exit $[$ret ? 1 : 0]
## submitted with:
# sbatch --export=PATH  -p RM-shared  --open-mode=append -e asr_stats_raw_en_char_espnet/logdir/q/stats.log -o asr_stats_raw_en_char_espnet/logdir/q/stats.log --array 1-4 /ocean/projects/cis210027p/jzmo/work/espnet/egs2/librispeech/hubert_asr1/asr_stats_raw_en_char_espnet/logdir/q/stats.sh >>asr_stats_raw_en_char_espnet/logdir/q/stats.log 2>&1
