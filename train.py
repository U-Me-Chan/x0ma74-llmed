import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model
from datasets import load_dataset

# Максимальная длина токенов (можно поменять)
MAX_LENGTH = 256

# 🔹 Загружаем базовую модель и токенизатор
model_name = "OpenBuddy/openbuddy-mistral-7b-v13"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,   # Используем bf16 вместо 8-bit
    low_cpu_mem_usage=True,         # Оптимизация памяти для Mac
    device_map={"": 0},             # Принудительно загружаем на GPU/CPU
    offload_folder="offload"        # Если VRAM не хватает
)

# model.to("cpu")

# 🔹 Загружаем датасет.
# Предполагается, что файл dataset.jsonl содержит строки вида:
# {"text": "съешь ещё этих французских булок да выпей чаю"}
dataset = load_dataset("json", data_files="dataset.jsonl", split="train")

# Функция для токенизации текста
def preprocess_function(examples):
    # Токенизируем поле "text" без использования return_tensors,
    # чтобы dataset.map корректно обработал данные.
    tokenized_output = tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH
    )
    # Для задачи causal LM метки (labels) равны input_ids.
    tokenized_output["labels"] = tokenized_output["input_ids"].copy()
    return tokenized_output

# Применяем токенизацию ко всему датасету и удаляем исходное поле "text"
tokenized_dataset = dataset.map(preprocess_function, batched=True, remove_columns=["text"])

# Используем data collator, специально предназначенный для языкового моделирования
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# 🔹 Настройка LoRA
lora_config = LoraConfig(
    r=12,                 # default: 8
    lora_alpha=20,        # default: 16
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="all",           # default: none
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)

# 🔹 Настройки обучения
training_args = TrainingArguments(
    output_dir="./stage-1-finetuned",
    per_device_train_batch_size=1,             # уменьшенный размер батча для Mac
    gradient_accumulation_steps=8,             # имитируем больший батч
    num_train_epochs=3,
    save_steps=500,
    logging_dir="./logs",
    fp16=False,                              # на Mac fp16 может давать ошибки; используем bf16
    bf16=True,                               # bf16 – оптимальный вариант для Apple Silicon
    optim="adamw_torch",
    remove_unused_columns=False
)

# Создаем Trainer для обучения
trainer = Trainer(
    model=model,
    train_dataset=tokenized_dataset,
    args=training_args,
    data_collator=data_collator,
)

# 🔹 Запуск обучения
trainer.train()
