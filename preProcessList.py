import numpy as np
import os
import librosa
from tqdm import tqdm
import json

tqdm.pandas()

DATASET_PATH = "datasets03"
JSON_PATH = "library.json"
SAMPLE_RATE = 22050
TRACK_DURATION = 4  # measured in seconds
SAMPLES_PER_TRACK = SAMPLE_RATE * TRACK_DURATION
num_segment_list = [1, 2, 3, 4, 5, 6, 7, 8]
num_mfcc = 13
n_fft = 2048
hop_length = 512

library = {
    "segment": [],
    "data": []
}

for ns in num_segment_list:
    data = {
        "mapping": [],
        "labels": [],
        "mfcc": [],
    }

    samples_per_segment = int(SAMPLES_PER_TRACK / ns)
    print("segment sayısı", ns, " işleniyor.")

    # loop through all genre sub-folder
    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(DATASET_PATH)):

        # ensure we're processing a genre sub-folder level
        if dirpath is not DATASET_PATH:

            # save genre label (i.e., sub-folder name) in the mapping
            semantic_label = dirpath.split("/")[-1]
            data["mapping"].append(semantic_label)
            print("\nİşlenen klasör: {}".format(semantic_label))

            # process all audio files in genre sub-dir
            filenum = 0
            for f in filenames:
                filenum = filenum + 1
                # load audio file
                file_path = os.path.join(dirpath, f)
                signal, sample_rate = librosa.load(file_path, sr=SAMPLE_RATE)

                # process all segments of audio file

                for d in range(ns):
                    # calculate start and finish sample for current segment
                    start = samples_per_segment * d
                    finish = start + samples_per_segment

                    # extract mfcc
                    mfcc = librosa.feature.mfcc(y=signal[start:finish], sr=sample_rate, n_mfcc=num_mfcc, n_fft=n_fft,
                                                hop_length=hop_length)
                    mfcc = mfcc.T
                    # store only mfcc feature with expected number of vectors
                    data["mfcc"].append(mfcc.tolist())
                    data["labels"].append(i - 1)

                    # print("ns: ", ns, "dosya:{}, segment:{}".format(filenum, d + 1))
    tensor_shape = np.array(data["mfcc"]).shape
    print("Tensör Boyutu: ", tensor_shape)
    # print("ns: ", ns, "{} dosya, {} segment".format(int(tensor_shape[0] / ns), tensor_shape[0]))
    library["data"].append(data)
    library["segment"].append(ns)
    # save MFCCs to json file
    with open(JSON_PATH, "w") as fp:
        json.dump(library, fp, indent=4)
