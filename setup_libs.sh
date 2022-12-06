pip install -r requirements.txt
git lfs install
python -c "import torch; print(torch.cuda.is_available())"
git config --global credential.helper store
huggingface-cli login
huggingface-cli repo create $1
git clone https://huggingface.co/Zaid/$1
cd $1
cp ../run_speech_recognition_seq2seq_streaming.py .
cp ../run.sh .