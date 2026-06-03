# Извлечение коллокатов (logDice) для двух подкорпусов

import pandas as pd
import re
import math
from collections import defaultdict

def load_stopwords(filepath='stopwords.txt'):
    with open(filepath, 'r', encoding='utf-8') as f:
        return {line.strip().lower() for line in f if line.strip()}

def clean_for_collocates(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def extract_collocates(texts, target_phrase, stop_words, window=5, min_freq=5, top_n=20):
    target_tokens = target_phrase.lower().split()
    collocate_freq = defaultdict(int)
    target_total = 0
    for text in texts:
        words = text.split()
        if len(target_tokens) == 1:
            positions = [i for i, w in enumerate(words) if w == target_tokens[0]]
        else:
            positions = []
            for i in range(len(words)-1):
                if words[i]==target_tokens[0] and words[i+1]==target_tokens[1]:
                    positions.append(i)
        target_total += len(positions)
        for pos in positions:
            start = max(0, pos - window)
            if len(target_tokens) == 1:
                end = min(len(words), pos + window + 1)
                for i in range(start, end):
                    if i == pos: continue
                    w = words[i]
                    if w not in stop_words and len(w)>2:
                        collocate_freq[w] += 1
            else:
                end = min(len(words), pos + window + 2)
                for i in range(start, end):
                    if i == pos or i == pos+1: continue
                    w = words[i]
                    if w not in stop_words and len(w)>2:
                        collocate_freq[w] += 1
    result = []
    R_total = target_total if target_total>0 else 1
    for word, O in collocate_freq.items():
        if O < min_freq: continue
        C = collocate_freq[word]
        logdice = 14 + math.log2((2*O)/(C+R_total)) if (C+R_total)>0 else 0
        result.append((word, O, logdice))
    result.sort(key=lambda x: x[2], reverse=True)
    return result[:top_n]

def print_table(collocates, phrase):
    print(f"\nтоп-20 коллокатов для '{phrase}' (logDice, окно 5l/5r):")
    print(f"{'№':<3} {'коллокат':<20} {'freq':<8} {'logDice':<8}")
    print('-'*45)
    for i,(w,f,ld) in enumerate(collocates,1):
        print(f"{i:<3} {w:<20} {f:<8} {ld:.2f}")

def main():
    stop_words = load_stopwords('stopwords.txt')
    df1 = pd.read_csv('coronavirus_subcorpus.csv')
    print(f"Загружено статей (coronavirus): {len(df1)}")
    texts1 = df1['clean_article'].astype(str).apply(clean_for_collocates).tolist()
    coll1 = extract_collocates(texts1, 'coronavirus', stop_words, window=5, min_freq=5)
    print_table(coll1, 'coronavirus')
    df2 = pd.read_csv('chinese_virus_subcorpus.csv')
    print(f"\nЗагружено статей (chinese virus): {len(df2)}")
    texts2 = df2['clean_article'].astype(str).apply(clean_for_collocates).tolist()
    coll2 = extract_collocates(texts2, 'chinese virus', stop_words, window=5, min_freq=2)
    print_table(coll2, 'chinese virus')

if __name__ == '__main__':
    main()
