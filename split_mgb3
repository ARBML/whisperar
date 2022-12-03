import soundfile as sf
import os

os.makedirs("dataset", exist_ok=True)
archive_path = "test"
wav_dir = os.path.join(archive_path, "wav")
segments_file = os.path.join(archive_path, "text.all")
with open(segments_file, "r", encoding="utf-8") as f:
    for _id, line in enumerate(f):
        segment = line.split(" ")[0]
        text = " ".join(line.split(" ")[1:])
        wav_name, _, time = segment.split("_")
        time = time.replace("seg-", "")
        start, stop = time.split(":")
        start = int(int(start) / 100 * 16_000)
        stop = int(int(stop) / 100 * 16_000)
        wav_path = os.path.join(wav_dir, wav_name + ".wav")
        sound, _ = sf.read(wav_path, start=start, stop=stop)
        sf.write(f"dataset/{segment}.wav", sound, 16_000)
        open(f"dataset/{segment}.txt", "w").write(text)
