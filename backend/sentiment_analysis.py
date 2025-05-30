import requests
from bs4 import BeautifulSoup
from transformers import pipeline
import re

def preprocess_text(text):
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^A-Za-z0-9À-ÿ.,;!?() ]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def fetch_news_from_source(url, parser, tag, attr_key, attr_value):
    headlines = []
    try:
        r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=7)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, parser)
        for tag_item in soup.find_all(tag, {attr_key: attr_value}):
            text = preprocess_text(tag_item.get_text())
            if len(text) > 30:
                headlines.append(text)
    except Exception as e:
        print(f"Erro ao buscar em {url}: {e}")
    return headlines

def enhanced_sentiment_analysis(ticker):
    sentiment_pipeline = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")

    sources = [
        # (URL, parser, tag, attribute_key, attribute_value)
        (f"https://valor.globo.com/busca/?q={ticker}", "html.parser", "a", "class", "feed-post-link"),
        (f"https://www.infomoney.com.br/?s={ticker}", "html.parser", "a", "class", "hl-title"),
        (f"https://exame.com/?s={ticker}", "html.parser", "a", "class", "title"),
        (f"https://br.investing.com/search/?q={ticker}", "html.parser", "a", "class", "title"),
        (f"https://busca.uol.com.br/?q={ticker}", "html.parser", "a", "class", "thumb-caption"),
        (f"https://g1.globo.com/busca/?q={ticker}", "html.parser", "a", "class", "widget--info__text-container"),
        (f"https://busca.estadao.com.br/?q={ticker}", "html.parser", "a", "class", "resultado"),
        (f"https://br.financas.yahoo.com/lookup?s={ticker}", "html.parser", "a", "class", "Fz(16px)"), # adaptável
        (f"https://www.seudinheiro.com/?s={ticker}", "html.parser", "a", "class", "post-title"),
        (f"https://www.suno.com.br/noticias/?s={ticker}", "html.parser", "a", "class", "entry-title"),
    ]

    all_headlines = []

    for url, parser, tag, attr_key, attr_value in sources:
        headlines = fetch_news_from_source(url, parser, tag, attr_key, attr_value)
        print(f"{len(headlines)} notícias de: {url}")
        all_headlines.extend(headlines)

    if not all_headlines:
        return {
            "final_sentiment": "neutro",
            "resume_sentiment": "Nenhuma notícia relevante encontrada.",
            "headlines_analyzed": []
        }

    results = sentiment_pipeline(all_headlines[:50])

    transformer_scores = []
    categorized = {"positivo": [], "neutro": [], "negativo": []}

    for i, res in enumerate(results):
        label = res['label']
        stars = int(label.split()[0])
        score = res['score']
        headline = all_headlines[i]
        if stars <= 2:
            transformer_scores.append(-score)
            categorized["negativo"].append(headline)
        elif stars == 3:
            transformer_scores.append(0)
            categorized["neutro"].append(headline)
        else:
            transformer_scores.append(score)
            categorized["positivo"].append(headline)

    avg_score = sum(transformer_scores) / len(transformer_scores)

    if avg_score > 0.15:
        final_sentiment = "fortemente positivo"
    elif avg_score > 0.05:
        final_sentiment = "positivo"
    elif avg_score < -0.15:
        final_sentiment = "fortemente negativo"
    elif avg_score < -0.05:
        final_sentiment = "negativo"
    else:
        final_sentiment = "neutro"

    summary = (
        f"Analisando {len(all_headlines)} manchetes de diversas fontes sobre '{ticker}', "
        f"o sentimento geral é **{final_sentiment}**.\n\n"
        f"- Positivas: {len(categorized['positivo'])}\n"
        f"- Neutras: {len(categorized['neutro'])}\n"
        f"- Negativas: {len(categorized['negativo'])}\n"
    )

    return {
        "final_sentiment": final_sentiment,
        "transformer_score": round(avg_score, 4),
        "headlines_analyzed": len(all_headlines),
        "resume_sentiment": summary,
        "top_positive": categorized["positivo"][:3],
        "top_negative": categorized["negativo"][:3]
    }