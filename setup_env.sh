cd ~
sudo add-apt-repository -y ppa:jonathonf/ffmpeg-4
sudo apt update
sudo apt install -y ffmpeg
sudo apt-get install git-lfs
env_name=whisper
python3 -m venv $env_name
echo "source ~/$env_name/bin/activate" >> ~/.bashrc
cd whisper_sprint
bash