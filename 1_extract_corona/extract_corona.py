# Обработка корпуса All the News

import pandas as pd
import re
import zipfile
import os
from google.colab import drive

drive.mount('/content/drive')

# 1. Распаковка archive.zip
zip_path = '/content/drive/MyDrive/archive.zip'
extract_dir = '/content/all_the_news'

if os.path.exists(zip_path):
    print("Распаковка архива...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)
    print("Архив распакован.")
else:
    print(f"Файл не найден: {zip_path}")

# 2. Поиск csv файла
csv_files = [f for f in os.listdir(extract_dir) if f.endswith('.csv')]
if not csv_files:
    raise Exception("CSV не найден")
csv_path = os.path.join(extract_dir, csv_files[0])
print(f"CSV найден: {csv_path}")

# 3. Функция очистки для vader
def clean_text_for_vader(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r'<.*?>', ' ', text)
    text = re.sub(r'http\S+|www\S+|https\S+', ' ', text)
    text = re.sub(r'\S+@\S+', ' ', text)
    text = re.sub(r'view the discussion thread', ' ', text, flags=re.IGNORECASE)
    text = re.sub(r'the hill \d+ k street', ' ', text, flags=re.IGNORECASE)
    text = re.sub(r'capitol hill publishing corp', ' ', text, flags=re.IGNORECASE)
    text = re.sub(r'news communications inc', ' ', text, flags=re.IGNORECASE)
    text = re.sub(r'advertisement|subscribe now|sign up|newsletter', ' ', text, flags=re.IGNORECASE)
    text = re.sub(r'privacy policy|cookies|sponsored|all rights reserved', ' ', text, flags=re.IGNORECASE)
    text = re.sub(r'â€™|â€œ|â€\x9d|â€"|â„¢|Â', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# 4. Обработка по чанкам
chunk_size = 50000
start_date = '2019-12-01'

print("Обработка CSV по чанкам")
filtered_data = []

for i, chunk in enumerate(pd.read_csv(csv_path, chunksize=chunk_size, low_memory=False)):
    if 'date' not in chunk.columns or 'article' not in chunk.columns:
        continue
    chunk['date'] = pd.to_datetime(chunk['date'], errors='coerce')
    chunk = chunk[chunk['date'] >= start_date]
    if chunk.empty:
        continue
    mask = (
        chunk['article'].str.contains('coronavirus', case=False, na=False, regex=False) |
        chunk['article'].str.contains('chinese virus', case=False, na=False, regex=False)
    )
    chunk = chunk[mask]
    if chunk.empty:
        continue
    chunk['clean_article'] = chunk['article'].apply(clean_text_for_vader)
    chunk = chunk[chunk['clean_article'].str.len() > 200]
    if 'publication' in chunk.columns:
        chunk = chunk[['date', 'publication', 'clean_article']]
    else:
        chunk = chunk[['date', 'clean_article']]
    filtered_data.append(chunk)
    print(f"Обработано чанков: {i+1}", end='\r')

if not filtered_data:
    raise Exception("Не найдено ни одной подходящей статьи")

df_filtered = pd.concat(filtered_data, ignore_index=True)
df_filtered = df_filtered.drop_duplicates(subset=['clean_article'])

print(f"\nОтфильтровано статей: {len(df_filtered):,}")

# Деление на подкорпусы
chinese_mask = df_filtered['clean_article'].str.contains(r'(?i)\bchinese virus\b', regex=True, na=False)
corona_mask = df_filtered['clean_article'].str.contains(r'(?i)\bcoronavirus\b', regex=True, na=False)
chinese_df = df_filtered[chinese_mask].copy()
corona_df = df_filtered[corona_mask & (~chinese_mask)].copy()

print(f"\nПодкорпус 'coronavirus': {len(corona_df):,} статей")
print(f"Подкорпус 'Chinese virus': {len(chinese_df):,} статей")

# Подсчет токенов
corona_df['word_count'] = corona_df['clean_article'].str.split().str.len()
chinese_df['word_count'] = chinese_df['clean_article'].str.split().str.len()
print(f"\nТокенов в 'coronavirus': {corona_df['word_count'].sum():,}")
print(f"Токенов в 'Chinese virus': {chinese_df['word_count'].sum():,}")

# Топ-10 изданий
if 'publication' in corona_df.columns:
    print("\nтоп-10 изданий (coronavirus):")
    print(corona_df['publication'].value_counts().head(10).to_string())
if 'publication' in chinese_df.columns:
    print("\nтоп-10 изданий (Chinese virus):")
    print(chinese_df['publication'].value_counts().head(10).to_string())

# Сохранение файлов
with open('corpus_coronavirus.txt', 'w', encoding='utf-8') as f:
    for text in corona_df['clean_article']:
        f.write(text + '\n\n')
with open('corpus_chinese_virus.txt', 'w', encoding='utf-8') as f:
    for text in chinese_df['clean_article']:
        f.write(text + '\n\n')
corona_df.to_csv('coronavirus_subcorpus.csv', index=False)
chinese_df.to_csv('chinese_virus_subcorpus.csv', index=False)

print("\nГотово: txt и csv файлы сохранены")
