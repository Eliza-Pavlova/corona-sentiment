# Сентимент-анализ подкорпусов и коллокатов (VADER)

import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

analyzer = SentimentIntensityAnalyzer()

def get_sentiment(text):
    if not isinstance(text, str):
        return {'compound': 0, 'pos': 0, 'neu': 0, 'neg': 0}
    return analyzer.polarity_scores(text)

def analyze_subcorpus(df, name):
    print(f"\nанализ подкорпуса '{name}'...")
    sentiments = [get_sentiment(text) for text in df['clean_article']]
    df['compound'] = [s['compound'] for s in sentiments]
    avg = df['compound'].mean()
    median = df['compound'].median()
    std = df['compound'].std()
    print(f"  статей: {len(df):,}")
    print(f"  средний compound: {avg:.4f}")
    print(f"  медиана: {median:.4f}")
    print(f"  стандартное отклонение: {std:.4f}")
    if avg > 0.05:
        print("  интерпретация: скорее позитивный")
    elif avg < -0.05:
        print("  интерпретация: скорее негативный")
    else:
        print("  интерпретация: нейтральный")
    return avg

def analyze_collocates(collocates_list, name):
    print(f"\nанализ коллокатов '{name}':")
    results = []
    for word in collocates_list:
        scores = analyzer.polarity_scores(word)
        results.append({
            'слово': word,
            'compound': scores['compound'],
            'pos': scores['pos'],
            'neu': scores['neu'],
            'neg': scores['neg']
        })
    df = pd.DataFrame(results)
    print(df[['слово', 'compound', 'pos', 'neg']].to_string(index=False))
    print(f"средний compound: {df['compound'].mean():.4f}")
    return df

# Списки коллокатов (из результатов второго этапа)
corona_collocates = [
    'outbreak', 'cases', 'spread', 'pandemic', 'said', 'china', 'novel',
    'impact', 'trump', 'response', 'people', 'confirmed', 'tested',
    'economic', 'global', 'positive', 'health', 'first', 'reported', 'crisis'
]

chinese_collocates = [
    'covid', 'coronavirus', 'trump', 'calling', 'virus', 'wrong', 'term',
    'said', 'china', 'use', 'wuhan', 'using', 'hit', 'phrase', 'house',
    'president', 'racist', 'defends', 'combat', 'production'
]

def main():
    print("загрузка подкорпусов...")
    corona_df = pd.read_csv('coronavirus_subcorpus.csv')
    chinese_df = pd.read_csv('chinese_virus_subcorpus.csv')
    print(f"загружено: {len(corona_df):,} статей (coronavirus), {len(chinese_df)} статей (chinese virus)")

    corona_avg = analyze_subcorpus(corona_df, 'coronavirus')
    chinese_avg = analyze_subcorpus(chinese_df, 'chinese virus')

    print("\nсравнение подкорпусов:")
    print(f"разница (chinese virus - coronavirus): {chinese_avg - corona_avg:.4f}")
    print(f"→ подкорпус 'chinese virus' негативнее на {abs(chinese_avg - corona_avg):.4f}")

    analyze_collocates(corona_collocates, 'coronavirus')
    analyze_collocates(chinese_collocates, 'chinese virus')

if __name__ == '__main__':
    main()
