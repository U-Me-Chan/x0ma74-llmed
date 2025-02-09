import json

# Имя входного и выходного файла
input_file = "raw_texts.txt"  # Обычный текстовый файл с примерами (по одной строке)
output_file = "dataset.jsonl"  # Готовый датасет в формате JSONL

# Базовые инструкции для обучения (можно кастомизировать)
instructions = [
    "Сгенерируй ответ в этом стиле:"
]

# Читаем текстовый файл
with open(input_file, "r", encoding="utf-8") as f:
    texts = [line.strip() for line in f.readlines() if line.strip()]

# Создаём JSONL-файл
with open(output_file, "w", encoding="utf-8") as f:
    for text in texts:
        instruction = instructions[len(text) % len(instructions)]  # Выбираем случайную инструкцию
        data = {
            "messages": [
                {"role": "user", "content": instruction},
                {"role": "assistant", "content": text}
            ]
        }
        json.dump(data, f, ensure_ascii=False)
        f.write("\n")

print(f"✅ Датасет сохранён в {output_file} ({len(texts)} примеров)")
