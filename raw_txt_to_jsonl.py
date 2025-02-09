import json

# Имя входного и выходного файла
input_file = "raw_texts.txt"   # Обычный текстовый файл с примерами (по одной строке)
output_file = "dataset.jsonl"  # Готовый датасет в формате JSONL

# Читаем текстовый файл
with open(input_file, "r", encoding="utf-8") as f:
    texts = [line.strip() for line in f if line.strip()]

# Создаём JSONL-файл
with open(output_file, "w", encoding="utf-8") as f:
    for text in texts:
        # Каждый пример сохраняется как один объект с ключом "text"
        data = {
            "text": text
        }
        json.dump(data, f, ensure_ascii=False)
        f.write("\n")

print(f"✅ Датасет сохранён в {output_file} ({len(texts)} примеров)")
