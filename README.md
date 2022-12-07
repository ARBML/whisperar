# whisper_sprint

## training 

```bash
git clone https://github.com/ARBML/whisper_sprint
cd whisper_sprint
```

Then setup the enviornment 

```bash
bash setup_env.sh
```

Then setup the libraries, this will install transofrmers, etc. and create a directory in the hub for training the model ... 

```bash
bash setup_lib.sh HF_USER_NAME MODEL_NAME
```

After that, you can run training by 

```
cd MODEL_NAME_HUB
bash run.sh
```

## evaluation 
```bash
python evaluate_models.py \
        --dataset_name google/fleurs\
        --model_name openai/whisper-tiny\
        --subset ar_eg\
        --hf_token hf_**
```
