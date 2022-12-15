python run_eval_whisper_streaming.py \
        --dataset google/fleurs \
        --model_id $1 \
        --split test \
        --config ar_eg \
        --batch_size="16" \
        --remove_diacritics="True" \
        --language="ar" \
        --device 0 \
        --streaming="False"