# Анализ тональности и коллокатов в корпусе новостей о COVID-19

Код для выпускной квалификационной работы.  
Сравнение терминов *coronavirus* и *Chinese virus* в англоязычном корпусе All the News.

## Структура

- `1_extract_corona/` — извлечение статей, формирование подкорпусов.
- `2_extract_collocates/` — извлечение коллокатов (logDice).
- `3_sentiment/` — сентимент-анализ подкорпусов и коллокатов (VADER).
- `data/` — папка для исходных данных (не загружается на GitHub).

## Данные

Исходный корпус (архив ~8 ГБ):  
[Ссылка на Google Диск](https://drive.google.com/file/d/1c1BpJLCTnSED12JUsw7wMC0TLe-hEdFp/view?usp=drive_link)

Перед запуском:
1. Скачайте `archive.zip` по ссылке.
2. Положите его в корень проекта или в `data/`.
3. Если запускаете в Colab, см. комментарии в скрипте.

## Установка

```bash
pip install -r requirements.txt
