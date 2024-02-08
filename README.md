# audiopipeline
This is a collection of audio/speech pipelines. So far focus more on data part 
(crawler, ETL). The intention is, some unpopular languages public audio dataset 
are quite limited. For example, Hokkien, although there is a quite good dataset 
[TAT](https://sites.google.com/nycu.edu.tw/fsw/home/tat-phase-i?authuser=0), but 
sadly they didn't open source it, which means you will meet difficuty on finding 
corpus when your are working on Taiwanese ASR model training.
 
Here this repo has following functionalities:
* Crawling audio and subtitle of Youtube videos, which can be used as speech models' 
  training corpus
* Automatically operate [NTUT's TTS service](http://tts001.iptcloud.net:8804/) (which 
  is a Hokkien TTS service hosted by NTUT) to generate some synthesis speech audios 
  based on given texts.


## Programs
### `bin/crawl_youtube_audio_and_cc_simple.py`
Crawling ASR dataset from Youtube audios.
```shell
python ./bin/crawl_youtube_audio_and_cc_simple.py ./demo_configs/crawl_youtube_audio_and_cc_simple.json
```

### `bin/gen_tw_hokkien_with_ntut_tts.py`
Generate Hokkien based on [NTUT's TTS service](http://tts001.iptcloud.net:8804/)
```shell
python bin/gen_tw_hokkien_with_ntut_tts.py ./demo_configs/gen_tw_hokkien_with_ntut_tts.json
```
