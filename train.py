import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
from transformers import DataCollatorWithPadding

# Максимальная длина токенов (можно поменять)
MAX_LENGTH = 256

# 🔹 Загружаем модель и токенизатор
model_name = "t-tech/T-lite-it-1.0"
tokenizer = AutoTokenizer.from_pretrained(model_name)
data_collator = DataCollatorWithPadding(tokenizer, padding=True)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,  # Используем bf16 вместо 8-bit
    low_cpu_mem_usage=True,        # Оптимизация памяти для Mac
    device_map={"": 0},            # Принудительно загружаем на GPU/CPU
    offload_folder="offload"       # Если VRAM не хватает
)

# 🔹 Загружаем датасет (JSONL с парами prompt-response)
dataset = load_dataset("json", data_files="dataset.jsonl")

# Функция для токенизации
def preprocess_function(examples):
    texts = []
    for message_list in examples["messages"]:
        if isinstance(message_list, list):
            user_messages = [msg["content"] for msg in message_list if msg.get("role") == "user"]
            texts.append(" ".join(user_messages))  # Объединяем в один текст

    # Токенизируем с padding и truncation
    tokenized_output = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt"
    )

    # Преобразуем тензоры в списки (если требуется) и добавляем labels как копию input_ids
    return {
        "input_ids": tokenized_output["input_ids"].tolist(),
        "attention_mask": tokenized_output["attention_mask"].tolist(),
        "labels": tokenized_output["input_ids"].tolist()
    }

# Применяем токенизацию
tokenized_datasets = dataset.map(preprocess_function, batched=True)

# Удаляем лишнюю колонку, чтобы DataCollator не пытался паддить "messages"
tokenized_datasets = tokenized_datasets.remove_columns(["messages"])

# 🔹 Настройка LoRA
lora_config = LoraConfig(
    r=8, lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)

# 🔹 Настройки обучения
training_args = TrainingArguments(
    output_dir="./stage-1-finetuned",
    per_device_train_batch_size=1,  # Mac имеет мало VRAM
    gradient_accumulation_steps=8,
    num_train_epochs=3,
    save_steps=500,
    logging_dir="./logs",
    fp16=False,  # fp16 может давать ошибки на Mac, лучше bf16
    bf16=True,   # bf16 - лучший вариант для Apple Silicon
    optim="adamw_torch",
    remove_unused_columns=False
)

trainer = Trainer(
    model=model,
    train_dataset=tokenized_datasets["train"],
    args=training_args,
    data_collator=data_collator
)

# 🔹 Запуск обучения
trainer.train()
