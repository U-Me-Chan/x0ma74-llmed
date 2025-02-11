import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

print("Укажите имя базовой модели и путь к дообученному чекпоинту адаптера")
base_model_name = "OpenBuddy/openbuddy-mistral-7b-v13"
checkpoint_path = "./stage-1-finetuned/checkpoint-230"

print("Загружаем токенизатор из базовой модели")
tokenizer = AutoTokenizer.from_pretrained(base_model_name)

print("Загружаем базовую модель (полная модель)")
model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    torch_dtype=torch.bfloat16,  # Используем bf16 вместо 8-bit
    low_cpu_mem_usage=True,        # Оптимизация памяти для Mac
    device_map={"": 0},            # Принудительно загружаем на GPU/CPU
    offload_folder="offload"       # Если VRAM не хватает
)

print("Загружаем LoRA-адаптер и применяем его к базовой модели")
model = PeftModel.from_pretrained(model, checkpoint_path)

print("Объединяем (сливаем) веса адаптера в базовую модель")
model = model.merge_and_unload()

print("Сохраняем получившуюся модель")
output_dir = "./stage-2-merged-model"
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
