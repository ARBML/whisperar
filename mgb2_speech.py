import datasets
import os


_DESCRIPTION = "MGB2 speech recognition dataset AR"
_HOMEPAGE = "https://arabicspeech.org/mgb2/"
_LICENSE = "MGB-2 License agreement"
_CITATION = """@misc{https://doi.org/10.48550/arxiv.1609.05625,
  doi = {10.48550/ARXIV.1609.05625},
  
  url = {https://arxiv.org/abs/1609.05625},
  
  author = {Ali, Ahmed and Bell, Peter and Glass, James and Messaoui, Yacine and Mubarak, Hamdy and Renals, Steve and Zhang, Yifan},
  
  keywords = {Computation and Language (cs.CL), FOS: Computer and information sciences, FOS: Computer and information sciences},
  
  title = {The MGB-2 Challenge: Arabic Multi-Dialect Broadcast Media Recognition},
  
  publisher = {arXiv},
  
  year = {2016},
  
  copyright = {arXiv.org perpetual, non-exclusive license}
}
"""
_DATA_ARCHIVE_ROOT = "archives/"
_DATA_URL = {
    "test": _DATA_ARCHIVE_ROOT + "mgb2_wav.test.tar.gz",
    "dev": _DATA_ARCHIVE_ROOT + "mgb2_wav.dev.tar.gz",
    "train": [_DATA_ARCHIVE_ROOT + f"mgb2_wav_{x}.train.tar.gz" for x in range(48)], # we have 48 archives
}
_TEXT_URL = {
    "test": _DATA_ARCHIVE_ROOT + "mgb2_txt.test.tar.gz",
    "dev": _DATA_ARCHIVE_ROOT + "mgb2_txt.dev.tar.gz",
    "train": _DATA_ARCHIVE_ROOT + "mgb2_txt.train.tar.gz",
}

class MGDB2Dataset(datasets.GeneratorBasedBuilder):
    def _info(self):
        return datasets.DatasetInfo(
        description=_DESCRIPTION,
        features=datasets.Features(
            {
                "path": datasets.Value("string"),
                "audio": datasets.Audio(sampling_rate=16_000),
                "text": datasets.Value("string"),
            }
        ),
        supervised_keys=None,
        homepage=_HOMEPAGE,
        license=_LICENSE,
        citation=_CITATION,
    )

    def _split_generators(self, dl_manager):
        wav_archive = dl_manager.download(_DATA_URL)
        txt_archive = dl_manager.download(_TEXT_URL)
        test_dir = "dataset/test"
        dev_dir = "dataset/dev"
        train_dir = "dataset/train"

        if dl_manager.is_streaming:
    
            return [
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "path_to_txt": test_dir + "/txt",
                    "path_to_wav": test_dir + "/wav",
                    "wav_files": [dl_manager.iter_archive(wav_archive['test'])],
                    "txt_files": dl_manager.iter_archive(txt_archive['test']),
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "path_to_txt": dev_dir + "/txt",
                    "path_to_wav": dev_dir + "/wav",
                    "wav_files": [dl_manager.iter_archive(wav_archive['dev'])],
                    "txt_files": dl_manager.iter_archive(txt_archive['dev']),
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "path_to_txt": train_dir + "/txt",
                    "path_to_wav": train_dir + "/wav",
                    "wav_files": [dl_manager.iter_archive(a) for a in wav_archive['train']],
                    "txt_files": dl_manager.iter_archive(txt_archive['train']),
                },
            ),
        ]
        else:
            return [
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "path_to_txt": test_dir + "/txt",
                    "path_to_wav": test_dir + "/wav",
                    "wav_files": [dl_manager.extract(wav_archive['test'])],
                    "txt_files": dl_manager.extract(txt_archive['test']),
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "path_to_txt": dev_dir + "/txt",
                    "path_to_wav": dev_dir + "/wav",
                    "wav_files": [dl_manager.extract(wav_archive['dev'])],
                    "txt_files": dl_manager.extract(txt_archive['dev']),
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "path_to_txt": train_dir + "/txt",
                    "path_to_wav": train_dir + "/wav",
                    "wav_files": [dl_manager.extract(a) for a in wav_archive['train']],
                    "txt_files": dl_manager.extract(txt_archive['train']),
                },
            ),
        ]


    
    def _generate_examples(self, path_to_txt, path_to_wav, wav_files, txt_files):
        """ 
        This assumes that the text directory alphabetically precedes the wav dir
        The file names for wav and text seem to match and are unique
        We can use them for the dictionary matching them
        """
        examples = {}
        id_ = 0
        # need to prepare the transcript - wave map
        for item in txt_files:
            if type(item) is tuple:
                # iter_archive will return path and file
                path, f = item
                txt = f.read().decode(encoding="utf-8").strip()
            else:
                # extract will return path only
                path = item
                with open(path, encoding="utf-8") as f:
                    txt = f.read().strip()

            if path.find(path_to_txt) > -1:
                # construct the wav path
                # which is used as an identifier
                wav_path = os.path.split(path)[1].replace("_utf8", "").replace(".txt", ".wav").strip()
                
                examples[wav_path] = {
                    "text": txt,
                    "path": wav_path,
                }

        for wf in wav_files:
            for item in wf:
                if type(item) is tuple:
                    path, f = item
                    wav_data = f.read()
                else:
                    path = item
                    with open(path, "rb") as f:
                        wav_data = f.read()
                if path.find(path_to_wav) > -1:
                    wav_path = os.path.split(path)[1].strip()
                    audio = {"path": path, "bytes": wav_data}
                    yield id_, {**examples[wav_path], "audio": audio}
                    id_ += 1
