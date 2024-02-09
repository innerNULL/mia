# audiopipeline
This is a collection of audio/speech pipelines. So far focus more on data part 
(crawler, ETL). The intention is, some unpopular languages' public audio dataset 
are quite limited. For example, Hokkien, although there is a quite good dataset 
[TAT](https://sites.google.com/nycu.edu.tw/fsw/home/tat-phase-i?authuser=0), but 
sadly they didn't open source it, which means you will meet difficulty on finding 
corpus when you are working on Taiwanese ASR model training.
 
Here this repo has following functionalities:
* Crawling audio and subtitle of Youtube videos, which can be used as speech models' 
  training corpus
* Automatically operate [NTUT's TTS service](http://tts001.iptcloud.net:8804/) (which 
  is a Hokkien TTS service hosted by NTUT) to generate some synthesis speech audios 
  based on given texts.


## Programs
### Build Environments
Please first install [FFmpeg](https://ffmpeg.org/download.html) and then build your 
python environments with:
```shell
python3 -m venv ./_venv --copies
source ./_venv/bin/activate
python -m pip install -r requirements.txt
# deactivate
```

**NOTE**
If you are not use `python -m venv`, then please make sure you've configed `PATH` 
to make which include `youtube-dl` bin. On Linux, if you use globle Python to 
install dependencies, then `youtube-dl` should be `~/.local/bin/youtube-dl`, if 
that's the case of you, then you do not need do anything since there is a 
simple automatically `youtube-dl` seeking mechanism to handle this case.

### `bin/crawl_youtube_audio_and_cc_simple.py`
Here is an example usage:
```shell
python ./bin/crawl_youtube_audio_and_cc_simple.py ./demo_configs/crawl_youtube_audio_and_cc_simple.json
```

This will crawling ASR dataset from Youtube audios, after crawling 
task finished, you should get a folder with following structure:
```
./_crawl_youtube_audio_and_cc_simple/
├── dataset
└── raw
```
The `raw` is the raw audio/subtitle data crawled from Youtube, and the dataset 
is generated after following steps:
* Crawling raw audio and subtitle files.
* Chunking subtitle according timestamp blocks in it.
* Merging adjacent subtitle chunks, the definition of 'adjacent' means current 
  chunk's start time is same with previous chunk's end time.
* Chunking audio into audio chunks according chunked & merged subtitle chunks' time scope.
* Dumping metadata of chunked audios.

Here is the the structure of `raw` sub-directory:
```
./_crawl_youtube_audio_and_cc_simple/raw/
├── OAjS5meBURk.mp3
├── OAjS5meBURk.zh-TW.vtt
├── kIMWtz9y8M8.mp3
└── kIMWtz9y8M8.zh-TW.vtt
```
And here is the structure of `dataset` sub-directory:
```
./_crawl_youtube_audio_and_cc_simple/dataset/
├── OAjS5meBURk_part0.mp3
├── OAjS5meBURk_part1.mp3
├── OAjS5meBURk_part10.mp3
├── OAjS5meBURk_part100.mp3
├── ...
├── kIMWtz9y8M8_part96.mp3
├── kIMWtz9y8M8_part97.mp3
├── kIMWtz9y8M8_part98.mp3
├── kIMWtz9y8M8_part99.mp3
└── metadata.jsonl
```
The `metadata.jsonl` is in following format:
```
{"transcript": "這陣子我認真思考過 總算想通了 我打算離開這裡 重新規劃新人生", "path": "/_crawl_youtube_audio_and_cc_simple/dataset/OAjS5meBURk_part0.mp3"}
{"transcript": "俊杰 你這麼做是不是因為 我和惠婷的關係", "path": "/_crawl_youtube_audio_and_cc_simple/dataset/OAjS5meBURk_part1.mp3"}
{"transcript": "安康 別誤會", "path": "/_crawl_youtube_audio_and_cc_simple/dataset/OAjS5meBURk_part2.mp3"}
...
```

### `bin/gen_tw_hokkien_with_ntut_tts.py`
Generate Hokkien based on [NTUT's TTS service](http://tts001.iptcloud.net:8804/)
```shell
python ./bin/gen_tw_hokkien_with_ntut_tts.py ./demo_configs/gen_tw_hokkien_with_ntut_tts.json
```
This is just a casual one, not quite robust.

## Corpus
### Youtube Videos with Subtitle
* Taiwanese Hokkien
    * Business Related
        * Dramas
            * ~[再见阿郎台语](https://www.youtube.com/playlist?list=PLKDgOPgC7DbTEiYcr5HXmCBYj6CTDEpgr)~
            * [市井豪門](https://www.youtube.com/results?search_query=%E5%B8%82%E4%BA%95%E8%B1%AA%E9%96%80)
    * Medical Related
        * Dramas
            * [醫世情 The Doctors](https://www.youtube.com/playlist?list=PLc8M1wVJOpHxOh8W_CsE0QVj6bU13mJg5)
            * [百歲仁醫 Dr. SP - The Centenarian Doctor](https://www.youtube.com/playlist?list=PLc8M1wVJOpHwKDgSCcbUS8KFZWD48Wt-R)
            * ~[台九線上的愛 Will Being Love](https://www.youtube.com/playlist?list=PLc8M1wVJOpHz3I6BFxNCAQYXjQs3rYt90)~
            * [大林學校 A Fool Like Me](https://www.youtube.com/playlist?list=PLc8M1wVJOpHzcXp3D15E3v2SAOJO9uqgD)
            * [慈悲的滋味 Taste of Compassion](https://www.youtube.com/playlist?list=PLc8M1wVJOpHxAHhq9lPS0To2zNc_iwVaI)
            * [最佳配角](https://www.youtube.com/playlist?list=PLc8M1wVJOpHyIAViBj6Vf_LwvvOJY4rhK)
            * [丘醫師的願望 Selfless Giving of Dr. Chiu](https://www.youtube.com/playlist?list=PLc8M1wVJOpHzH8922CpOGsdVSSjxvukgi)
            * [(白袍的約定)](https://www.youtube.com/playlist?list=PLc8M1wVJOpHzexPvfep4vqpdGFNpWGgjL)
            * [(烏陰天的好日子)](https://www.youtube.com/playlist?list=PLzgAweye8Ud6ZWZ2ikBx1ee2uYszHa0cp)
    * Life Related
        * Dramas
            * [頂坡角上的家 Home Away From Home](https://www.youtube.com/playlist?list=PLc8M1wVJOpHwCUcO0OUF6tdw6Ythw6FVq)
            * [若是來恆春 Coming to Hengchun](https://www.youtube.com/playlist?list=PLc8M1wVJOpHyBYJUZYTC8Tyy_7z4k9A9-)
            * [清風無痕 Life As It Is](https://www.youtube.com/playlist?list=PLc8M1wVJOpHzSQHus3ZxkRfLPKrDHK-A6)
            * [人生二十甲 My Father’s Words](https://www.youtube.com/playlist?list=PLc8M1wVJOpHxvv-rEzb-LpEWht_3kBqj8)
            * [幸福的滋味 The taste of blessed](https://www.youtube.com/playlist?list=PLc8M1wVJOpHzIRHjvMO4TSlZZsw7wm4NH)

* Taiwanese Mandarin
    * Medical Related
        * Dramas
            * [白色巨塔](https://www.youtube.com/playlist?list=PLZB1HSq1adjj9Nd7G7R3ylRSt06XIkTT1)
            * [白袍之恋White Robe of Love](https://www.youtube.com/playlist?list=PLzt2yjwjKLWutCvoaH-5HgsTewH7ZJ7D6)
