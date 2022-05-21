python run_clm.py \
  --model_name_or_path gpt2-chinese-cluecorpussmall \
  --train_file all.txt \
  --per_device_train_batch_size 6 \
  --do_train \
  --output_dir ./medical-clm
