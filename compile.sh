mkdir stage-3-result
python3 convert_hf_to_gguf.py ../stage-2-merged-model --outfile=../stage-3-result/x0ma74_GGUF_Q8_0.gguf --outtype=q8_0 --model-name x0ma74_GGUF_Q8_0
