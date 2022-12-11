python run_speech_recognition_seq2seq_mixed_mgb2.py \
--model_name_or_path="openai/whisper-small" \
--dataset_name="arbml/mgb2_speech" \
--dataset_config_name="ar" \
--language="Arabic" \
--train_split_name="train+validation" \
--eval_split_name="test" \
--model_index_name="Whisper Small Arabic" \
--max_steps="5000" \
--output_dir="./" \
--per_device_train_batch_size="16" \
--per_device_eval_batch_size="8" \
--logging_steps="25" \
--learning_rate="1e-5" \
--warmup_steps="500" \
--evaluation_strategy="steps" \
--eval_steps="1000" \
--save_strategy="steps" \
--save_steps="1000" \
--generation_max_length="225" \
--length_column_name="input_length" \
--max_duration_in_seconds="30" \
--text_column_name="text" \
--freeze_feature_encoder="False" \
--report_to="tensorboard" \
--metric_for_best_model="wer" \
--greater_is_better="False" \
--load_best_model_at_end \
--gradient_checkpointing \
--fp16 \
--overwrite_output_dir \
--optim="adamw_bnb_8bit" \
--do_train \
--do_eval \
--predict_with_generate \
--do_normalize_eval \
--use_auth_token \
--push_to_hub
