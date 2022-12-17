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
bash setup_libs.sh HF_USER_NAME MODEL_NAME
```

After that, you can run training by 

```
cd MODEL_NAME
bash run_mgb2.sh
```

You can also run with deepspeed wich allows running whisper-large v2 with batch size 32 on A100

```
bash run_mgb2_deepspeed.sh
```

## Evaluation

### Evaluation on Fleurs

```
bash run_eval_fleurs.sh MODEL_NAME
```

### Evaluation on Common Voice 11

```
bash run_eval_cv_11.sh MODEL_NAME
```

## Preparing the MGB2 data

While MGB2 dataset contains a richly transcribed speech dataset, the wav files were too lengthy to be used to train the whisper model. Therefore, we had to split the wave file and still maintain the correct correspondence with the transcribed text.

MGB2 provides and XML file corresponding to every wav file, which contains the transcribed sentences and the start and end time of each sentence in the recording. Using the `split_xml_mgb2.py`, we start with the xml file and split the lengthy wav files into smaller ones that are shorter than 30 seconds in length, as required to fine-tune whisper. The operation produced over 370K sentences with their corresponding wav files.

## Hosting on HuggingFace (Privately)

To host mgb2 at HF, at least 3 things need to happen:

1. Create the dataset repository on HF. This was created privately at arbml/mgb2_speech for the dataset
2. Data must be hosted somewhere or uploaded to HF repo
3. HF loading script must be written so the data can be integrated into the HF hub.


### Uploading the data

The dataset was >100Gb in size. HF utilizes git lfs to host large files. However, git lfs has a max limit of 5gb size for any file. Uploading over 370K individual files was also not feasible and caused issues with git. 
Therefore, the solution was to archive groups of wav files together into sequentially numbered archive files, such that the archive file is no bigger than 5GB. To achieve that, the wav files were grouped based on the first 2 letters of the file name. The naming scheme seems to use a base64 encoding. So, characters would be 0 to 9 or A to F. The files were grouped as follows:

| First 2 Letters  | Archive Number  |
|:-:|---|
| 00-05  |  0 |
|  06-09 |  1 |
|  0A-0F |  2 |
| 10-15  |  3 |
|  16-19 |  4 |
|  1A-1F |  5 |
| ...  |  ... |
| F0-F5  |  45 |
|  F6-F9 |  46 |
|  FA-FF |  47 |

Only the training data was split using this scheme, the test and validation data was smaller than 5GB when archived.

### HF Data Loading Script

The loading script determines the features of the data based on split and selected configuration. We had test, dev, and train split with a single language configuration. Using the _generate_example function, the script is used by GH to correctly produce the associated transcript and wav files. The function works as follows:

1. Go through all the entries in the archive containing the text transcripts and create a map where the name of the file (the 64base encoded one) is used as the key and the transcript at the value
2. Iterate through all the wav files in all the archive, and for every wav file, get the corresponding transcript from the map constructed in previous step (using the file name) and yield the wav file, transcript, and path to the wav file

