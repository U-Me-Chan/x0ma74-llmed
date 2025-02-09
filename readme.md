# xoma74 llm'ed project
## Штука, чтобы обхитрить Хому

### how to

1. clone this repo
```sh
git clone https://github.com/U-Me-Chan/x0ma74-llmed
```

2. cd into and clone llama.cpp repo
```sh
cd x0ma74-llmed
git clone https://github.com/ggerganov/llama.cpp
```

3. install python deps
```sh
pip3 install torch transformers peft numpy datasets safetensors tokenizers sentencepiece accelerate --break-system-packages
```

4. convert raw x0ma74's text from `raw_texts.txt` into dataset
```sh
python3 raw_txt_to_jsonl.py
```

5. start training process
```sh
python3 train.py
```

6. merge finetuned params with origin model
```sh
python3 merge.py
```

7. convert resulted fp16-model into GGUF file with q8-model
```sh
./compile.sh
```

???. optional: u can test, but i dont test this test script (it run too much time, so i just load GGUF file into LM Studio)
```sh
python3 test.py
```
