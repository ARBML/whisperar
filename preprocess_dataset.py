from datasets import load_dataset, DatasetDict
from transformers import WhisperFeatureExtractor
from transformers import WhisperTokenizer
from transformers import WhisperProcessor
from datasets import Audio
from transformers.models.whisper.english_normalizer import BasicTextNormalizer
from huggingface_hub import login

import argparse

my_parser = argparse.ArgumentParser()

my_parser.add_argument(
    "--model_name",
    "-model_name",
    type=str,
    action="store",
    default="openai/whisper-tiny",
)
my_parser.add_argument("--hf_token", "-hf_token", type=str, action="store")
my_parser.add_argument(
    "--dataset_name", "-dataset_name", type=str, action="store", default="google/fleurs"
)
my_parser.add_argument("--split", "-split", type=str, action="store", default="test")
my_parser.add_argument("--subset", "-subset", type=str, action="store")

args = my_parser.parse_args()

dataset_name = args.dataset_name
model_name = args.model_name
subset = args.subset
hf_token = args.hf_token
login(hf_token)
text_column = "sentence"
if dataset_name == "google/fleurs":
    text_column = "transcription"

do_lower_case = False
do_remove_punctuation = False

normalizer = BasicTextNormalizer()
processor = WhisperProcessor.from_pretrained(
    model_name, language="Arabic", task="transcribe"
)
dataset = load_dataset(dataset_name, subset, use_auth_token=True)

print(dataset)

feature_extractor = WhisperFeatureExtractor.from_pretrained(model_name)

tokenizer = WhisperTokenizer.from_pretrained(
    model_name, language="Arabic", task="transcribe"
)
dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))


def prepare_dataset(batch):
    # load and (possibly) resample audio data to 16kHz
    audio = batch["audio"]

    # compute log-Mel input features from input audio array
    batch["input_features"] = processor.feature_extractor(
        audio["array"], sampling_rate=audio["sampling_rate"]
    ).input_features[0]
    # compute input length of audio sample in seconds
    batch["input_length"] = len(audio["array"]) / audio["sampling_rate"]

    # optional pre-processing steps
    transcription = batch[text_column]
    if do_lower_case:
        transcription = transcription.lower()
    if do_remove_punctuation:
        transcription = normalizer(transcription).strip()

    # encode target text to label ids
    batch["labels"] = processor.tokenizer(transcription).input_ids
    return batch


dataset = dataset.map(prepare_dataset, remove_columns=dataset.column_names["train"])

login(hf_token)
print(
    f"pushing to arbml/{dataset_name.split('/')[-1]}_preprocessed_{model_name.split('/')[-1]}"
)
dataset.push_to_hub(
    f"arbml/{dataset_name.split('/')[-1]}_preprocessed_{model_name.split('/')[-1]}",
    private=True,
)
