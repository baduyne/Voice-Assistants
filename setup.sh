cd web
# mkdir -p models


# STT model 
gdown --fuzzy "https://drive.google.com/file/d/17gKoKGRddEiDMPTR4wrKHTqTjlx9DAns/view?usp=sharing" -O models/STT-ggml-model.bin

# TTS model 
gdown --fuzzy "https://drive.google.com/file/d/1oAQA9-OkMM73nUnsfrpXDGAXeUJB9l48/view?usp=sharing" -O models/TTS_model.onnx
