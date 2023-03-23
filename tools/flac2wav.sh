dir=$1

audio_files=`find $dir -name "*.flac"`
for audio_file in $audio_files; do
    new_audio_file=`echo $audio_file | sed 's/\.flac/\.wav/g'`
    ffmpeg -i $audio_file -acodec pcm_s16le -ac 1 -ar 16000 $new_audio_file -nostdin
done