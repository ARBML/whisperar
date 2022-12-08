pip install -r requirements.txt
git lfs install
python -c "import torch; print(torch.cuda.is_available())"
git config --global credential.helper store
huggingface-cli login
huggingface-cli repo create $2
git clone https://huggingface.co/$1/$2
cd $2
cp ../run_speech_recognition_seq2seq_streaming.py .
cp ../run_speech_recognition_seq2seq.py .
cp ../run.sh .
wget https://raw.githubusercontent.com/huggingface/community-events/main/whisper-fine-tuning-event/fine-tune-whisper-non-streaming.ipynb
