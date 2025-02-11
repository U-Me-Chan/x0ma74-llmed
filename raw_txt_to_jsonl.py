import json
from transformers import AutoTokenizer

# 🔹 Настроим токенизатор (можно заменить на свой)
model_name = "OpenBuddy/openbuddy-mistral-7b-v13"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 🔹 Лимит токенов
MAX_LENGTH = 256

# 🔹 Функция для разбиения текста на предложения и упаковки в batch
def split_text_into_batches(text, max_length):
    sentences = text.split(". ")  # Разбиваем по точкам
    current_batch = []
    batches = []

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        # Токенизируем текущее предложение
        tokenized_sentence = tokenizer(sentence, add_special_tokens=False)
        token_count = len(tokenized_sentence["input_ids"])

        # Проверяем, влезает ли оно в текущий batch
        current_tokens = sum(len(tokenizer(s, add_special_tokens=False)["input_ids"]) for s in current_batch)
        if current_tokens + token_count <= max_length:
            current_batch.append(sentence)
        else:
            # Сохраняем текущую строку и начинаем новую
            batches.append(". ".join(current_batch) + ".")
            current_batch = [sentence]

    # Добавляем последний кусок
    if current_batch:
        batches.append(". ".join(current_batch) + ".")

    return batches

# 🔹 Обрабатываем входной файл
input_file = "raw_texts.txt"
output_file = "dataset.jsonl"

with open(input_file, "r", encoding="utf-8") as f, open(output_file, "w", encoding="utf-8") as out_f:
    for line in f:
        line = line.strip()
        if not line:
            continue

        batches = split_text_into_batches(line, MAX_LENGTH)

        # Сохраняем в JSONL-формате
        for batch in batches:
            json.dump({"text": batch}, out_f, ensure_ascii=False)
            out_f.write("\n")

print(f"✅ Датасет сохранён в {output_file}")
