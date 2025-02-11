import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel  # Используем PeftModel для загрузки адаптера

# Указываем имя базовой модели и путь к чекпоинту с адаптером
base_model_name = "OpenBuddy/openbuddy-mistral-7b-v13"
checkpoint_path = "./stage-2-merged-model/checkpoint-230"

# Загружаем токенизатор из базовой модели
tokenizer = AutoTokenizer.from_pretrained(base_model_name)

# Загружаем базовую модель (полную, без адаптера)
model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    torch_dtype=torch.bfloat16,  # Используем bf16 вместо 8-bit
    low_cpu_mem_usage=True,        # Оптимизация памяти для Mac
    device_map={"": 0},            # Принудительно загружаем на GPU/CPU
    offload_folder="offload"       # Если VRAM не хватает
)

# Загружаем дообученный адаптер LoRA и применяем его к базовой модели
model = PeftModel.from_pretrained(model, checkpoint_path)

# Переводим модель в режим оценки
model.eval()

# Пример запроса
prompt = "Привет, Детектив Хэнк. Каков ответ на вопрос о жизни, Вселенной и всего такого? Можешь уточнить у Оракула."

# Токенизируем запрос и перемещаем данные на устройство
device = "cuda" if torch.cuda.is_available() else "cpu"
inputs = tokenizer(prompt, return_tensors="pt").to("mps")

# Генерируем ответ
with torch.no_grad():
    output = model.generate(**inputs, max_new_tokens=100, do_sample=True, temperature=0.7)

print("Prompt:")
print(prompt)
print("\n\n")
print("Response:")
# Декодируем и выводим результат
print(tokenizer.decode(output[0], skip_special_tokens=True))
