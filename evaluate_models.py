import torch
import librosa
from datasets import load_dataset, Audio
from transformers import WhisperProcessor, WhisperFeatureExtractor, WhisperTokenizer, WhisperForConditionalGeneration
from huggingface_hub import login
import argparse
from evaluate import load

my_parser = argparse.ArgumentParser()
# my_parser.add_argument("--pal", "-paths_as_labels", action="store_true")

my_parser.add_argument("--model_name", "-model_name", type=str, action="store", default = "openai/whisper-tiny")
my_parser.add_argument("--hf_token", "-hf_token", type=str, action="store")
my_parser.add_argument("--dataset_name", "-dataset_name", type=str, action="store", default = "google/fleurs")
my_parser.add_argument("--split", "-split", type=str, action="store", default = "test")
my_parser.add_argument("--subset", "-subset", type=str, action="store")

args = my_parser.parse_args()
try:
  login(args.hf_token)
except:
  raise(f"Can't login please set --hf_token {args.hf_token}")


dataset_name = args.dataset_name 
model_name = args.model_name
subset = args.subset
text_column = "sentence"
if dataset_name == "google/fleurs":
  text_column = "transcription"
  
print(f"Evaluating {args.model_name} on {args.dataset_name} [{subset}]")


feature_extractor = WhisperFeatureExtractor.from_pretrained(model_name)
model = WhisperForConditionalGeneration.from_pretrained(model_name)

test_dataset = load_dataset(dataset_name, subset, split=args.split, use_auth_token=True)
processor = WhisperProcessor.from_pretrained(model_name, language="Arabic", task="transcribe")
tokenizer = WhisperTokenizer.from_pretrained(model_name, language="Arabic", task="transcribe")
test_dataset = test_dataset.cast_column("audio", Audio(sampling_rate=16000))

# Preprocessing the datasets.
def prepare_dataset(batch):
    # load and resample audio data from 48 to 16kHz
    audio = batch["audio"]

    # compute log-Mel input features from input audio array 
    batch["input_features"] = feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]

    # encode target text to label ids 
    batch["labels"] = tokenizer(batch[text_column]).input_ids
    return batch

test_dataset = test_dataset.map(prepare_dataset)

model = model.to("cuda")
model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(language = "ar", task = "transcribe")

def map_to_result(batch):

  with torch.no_grad():
    input_values = torch.tensor(batch["input_features"], device="cuda").unsqueeze(0)
    pred_ids = model.generate(input_values)

  batch["pred_str"] = processor.batch_decode(pred_ids, skip_special_tokens = True)[0]
  batch["text"] = processor.decode(batch["labels"], skip_special_tokens = True)
  
  return batch
results = test_dataset.map(map_to_result)

wer = load("wer")
print("Test WER: {:.3f}".format(wer.compute(predictions=results["pred_str"], references=results["text"])))
