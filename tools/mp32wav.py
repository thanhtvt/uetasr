import multiprocessing
import os
import sys
from glob import glob


def convert_mp3_to_wav(mp3_path, wav_path):
    """
    Convert an MP3 file to a WAV file.
    :param mp3_path: path to the MP3 file
    :param wav_path: path to the output WAV file
    """
    os.system("sox {} -r 16000 -b 16 -c 1 {}".format(mp3_path, wav_path))


if __name__ == "__main__":
    data_dir = sys.argv[1]
    output_dir = sys.argv[2]
    num_workers = int(sys.argv[3])  # specify the number of worker processes
    # glob data_dir to find *.mp3 files
    files = glob(os.path.join(data_dir, "**/*.mp3"), recursive=True)
    # apply multiprocessing to convert mp3 to wav
    with multiprocessing.Pool(num_workers) as pool:
        for mp3_path in files:
            rel_path = os.path.relpath(mp3_path, data_dir)
            wav_path = os.path.join(output_dir, os.path.splitext(rel_path)[0] + ".wav")
            pool.apply_async(convert_mp3_to_wav, args=(mp3_path, wav_path))
        print(mp3_path)
        pool.close()
        pool.join()
