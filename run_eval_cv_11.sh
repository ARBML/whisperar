python run_eval_whisper_streaming.py \
        --dataset mozilla-foundation/common_voice_11_0 \
        --model_id $1 \
        --split test \
        --config="ar" \
        --batch_size="16" \
        --remove_diacritics \
        --language="ar" \
        --device 0