## Introduction
## Common Using Commands
### Start Local LLM Server
* `llama.cpp` Server
    * Build CPU/GPI Library and Server
    ```shell
    mkdir build
    cd build

    # CPU
    cmake ../
    make -j8

    # GPU
    cmake ../ -DLLAMA_CUBLAS=on -DCMAKE_CUDA_ARCHITECTURES=all-major
    make -j8
    ```
    * Download Model Files From [HuggingFace](https://huggingface.co/TheBloke/Llama-2-7B-GGUF) 
    * Run `CUDA_VISIBLE_DEVICES=1 ./bin/server -m ../models/llama-2-7b.Q8_0.gguf -c 2048 --main-gpu 3 --n-gpu-layers 50`

* `ollama` Server
    * Run `ollama serve` to start server.
    * Run `ollama run llama2` to register model `llama2` into server


## Programs
### Evaluation Zero-Shot Summarisation Performance with LLM Server
```
python ./bin/evaluation/llm/eval_zero_shot_text_summarisation.py ./demo_configs/evaluation/llm/eval_zero_shot_text_summarisation.llamacpp.json
python ./bin/evaluation/llm/eval_zero_shot_text_summarisation.py ./demo_configs/evaluation/llm/eval_zero_shot_text_summarisation.ollama.json
```
