# audiopipeline
This is a collection of audio/speech pipelines. So far focus more on data part 
(crawler, ETL). The intention is, some unpopular languages' public audio dataset 
are quite limited. For example, Hokkien, although there is a quite good dataset 
[TAT](https://sites.google.com/nycu.edu.tw/fsw/home/tat-phase-i?authuser=0), but 
sadly they didn't open source it, which means you will meet difficuty on finding 
corpus when you are working on Taiwanese ASR model training.
 
Here this repo has following functionalities:
* Crawling audio and subtitle of Youtube videos, which can be used as speech models' 
  training corpus
* Automatically operate [NTUT's TTS service](http://tts001.iptcloud.net:8804/) (which 
  is a Hokkien TTS service hosted by NTUT) to generate some synthesis speech audios 
  based on given texts.


## Programs
Before all you should build your pyton environments:
```shell
python3 -m venv ./_venv --copies
source ./_venv/bin/activate
python -m pip install -r requirements.txt
```

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

## Corpus
### Youtube Dramas with Subtitle
* Taiwanese Hokkien
    * Business Related
        * ~[再见阿郎台语](https://www.youtube.com/playlist?list=PLKDgOPgC7DbTEiYcr5HXmCBYj6CTDEpgr)~
        * [市井豪門](https://www.youtube.com/results?search_query=%E5%B8%82%E4%BA%95%E8%B1%AA%E9%96%80)
    * Medical Related
        * [大林學校 A Fool Like Me](https://www.youtube.com/playlist?list=PLc8M1wVJOpHzcXp3D15E3v2SAOJO9uqgD)
        * [慈悲的滋味 Taste of Compassion](https://www.youtube.com/playlist?list=PLc8M1wVJOpHxAHhq9lPS0To2zNc_iwVaI)
        * ~[白袍的約定](https://www.youtube.com/playlist?list=PLc8M1wVJOpHzexPvfep4vqpdGFNpWGgjL)~
        * ~[烏陰天的好日子](https://www.youtube.com/playlist?list=PLzgAweye8Ud6ZWZ2ikBx1ee2uYszHa0cp)~
    * Life Related
        * [頂坡角上的家 Home Away From Home](https://www.youtube.com/playlist?list=PLc8M1wVJOpHwCUcO0OUF6tdw6Ythw6FVq)

* Taiwanese Mandarin
    * Medical Related
        * [白色巨塔](https://www.youtube.com/playlist?list=PLZB1HSq1adjj9Nd7G7R3ylRSt06XIkTT1)
        * [白袍之恋White Robe of Love](https://www.youtube.com/playlist?list=PLzt2yjwjKLWutCvoaH-5HgsTewH7ZJ7D6)
