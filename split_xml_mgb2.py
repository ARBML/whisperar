from pathlib import Path 
import soundfile as sf
import xml.etree.ElementTree as ET

split = "train" # or "dev"

# set the following path to where you
# extracted the mgb2 archive
archive_path = Path("data/train")

wav_dir = archive_path / "wav"
segments_file = archive_path / "xml" / "utf8"
# output directories
output_wav_dir = archive_path / "dataset" / split /"wav"
output_txt_dir = archive_path / "dataset" / split /"txt"

# create directories for output datasets
output_wav_dir.mkdir(parents=True, exist_ok=True)
output_txt_dir.mkdir(parents=True, exist_ok=True)

# for all xml segments files under utf8 directory from archive
for s_file in segments_file.glob("*.xml"):
    tree = ET.parse(str(s_file))
    root = tree.getroot()
    head = root[0]
    segments = root[1][0]
    
    # get the name of the wav file form the recording tag
    for child in head:
        if child.tag == "recording":
            print(child.attrib)
            file_name = child.attrib.get("filename")

    # get the start and end times from the segment under segments tag
    # and join the text from each segment to construct the transcript
    for segment in segments:
        start_time = int(float(segment.attrib.get("starttime")) *16_000)
        end_time = int(float(segment.attrib.get("endtime")) * 16_000)

        text = " ".join([x.text for x in segment])


        # now store the meta data and the correctly sampled wav file in the correct
        # output directories
        wav_path = wav_dir / f"{file_name}.wav"
        sound, _ = sf.read(wav_path, start=start_time, stop=end_time)
        sf.write(output_wav_dir / f"{file_name}_seg{start_time}_{end_time}.wav", sound, 16_000)
        open(output_txt_dir / f"{file_name}_seg{start_time}_{end_time}.txt", "w").write(text)