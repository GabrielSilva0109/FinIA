import requests
from bs4 import BeautifulSoup
import re
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering

# Modelo BERT em português para sentimento
MODEL_NAME = "neuralmind/bert-base-portuguese-cased-sentiment"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
model.eval()

# Modelo para embeddings e clusterização de manchetes semelhantes
embedding_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

def preprocess_text(text):
    text = re.sub(r"http\S+", "", text)  # remove URLs
    text = re.sub(r"[^A-Za-z0-9À-ÿ.,;!?() ]+", " ", text)  # remove caracteres estranhos
    text = re.sub(r"\s+", " ", text).strip()  # remove espaços extras
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

def cluster_headlines(headlines, threshold=0.75):
    # Cria embeddings
    embeddings = embedding_model.encode(headlines, convert_to_tensor=False)
    embeddings = np.array(embeddings)
    # Cluster usando aglomerativo para agrupar similares
    clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=1-threshold, affinity='cosine', linkage='average')
    clustering.fit(embeddings)
    clusters = {}
    for idx, label in enumerate(clustering.labels_):
        clusters.setdefault(label, []).append(headlines[idx])
    # Escolhe um representante por cluster (a mais curta)
    representatives = [min(c, key=len) for c in clusters.values()]
    return representatives

def predict_sentiment(texts):
    # Divide em batches para evitar estouro de memória
    batch_size = 16
    sentiments = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=128)
        with torch.no_grad():
            outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=1).cpu().numpy()
        # Classes: 0=negativo,1=neutro,2=positivo
        preds = np.argmax(probs, axis=1)
        scores = np.max(probs, axis=1)
        sentiments.extend(zip(preds, scores))
    return sentiments

def enhanced_sentiment_analysis(ticker):
    sources = [
        (f"https://valor.globo.com/busca/?q={ticker}", "html.parser", "a", "class", "feed-post-link"),
        (f"https://www.infomoney.com.br/?s={ticker}", "html.parser", "a", "class", "hl-title"),
        (f"https://exame.com/?s={ticker}", "html.parser", "a", "class", "title"),
        (f"https://br.investing.com/search/?q={ticker}", "html.parser", "a", "class", "title"),
        (f"https://busca.uol.com.br/?q={ticker}", "html.parser", "a", "class", "thumb-caption"),
        (f"https://g1.globo.com/busca/?q={ticker}", "html.parser", "a", "class", "widget--info__text-container"),
        (f"https://busca.estadao.com.br/?q={ticker}", "html.parser", "a", "class", "resultado"),
        (f"https://br.financas.yahoo.com/lookup?s={ticker}", "html.parser", "a", "class", "Fz(16px)"),
        (f"https://www.seudinheiro.com/?s={ticker}", "html.parser", "a", "class", "post-title"),
        (f"https://www.suno.com.br/noticias/?s={ticker}", "html.parser", "a", "class", "entry-title"),
    ]

    all_headlines = []
    for url, parser, tag, attr_key, attr_value in sources:
        headlines = fetch_news_from_source(url, parser, tag, attr_key, attr_value)
        print(f"Buscadas {len(headlines)} notícias de: {url}")
        all_headlines.extend(headlines)

    if not all_headlines:
        return {
            "final_sentiment": "neutro",
            "resume_sentiment": "Nenhuma notícia relevante encontrada.",
            "headlines_analyzed": 0,
            "top_positive": [],
            "top_negative": []
        }

    # Agrupa manchetes similares para evitar repetição
    unique_headlines = cluster_headlines(all_headlines, threshold=0.75)
    print(f"Manchetes únicas após clusterização: {len(unique_headlines)}")

    predictions = predict_sentiment(unique_headlines)

    categorized = {"positivo": [], "neutro": [], "negativo": []}
    scores = []

    for i, (sent, score) in enumerate(predictions):
        headline = unique_headlines[i]
        if sent == 0:
            categorized["negativo"].append(headline)
            scores.append(-score)
        elif sent == 1:
            categorized["neutro"].append(headline)
            scores.append(0)
        else:
            categorized["positivo"].append(headline)
            scores.append(score)

    avg_score = np.mean(scores)

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
        f"Analisando {len(unique_headlines)} manchetes de diversas fontes sobre '{ticker}', "
        f"o sentimento geral é **{final_sentiment}**.\n\n"
        f"- Positivas: {len(categorized['positivo'])}\n"
        f"- Neutras: {len(categorized['neutro'])}\n"
        f"- Negativas: {len(categorized['negativo'])}\n"
    )

    return {
        "final_sentiment": final_sentiment,
        "transformer_score": round(avg_score, 4),
        "headlines_analyzed": len(unique_headlines),
        "resume_sentiment": summary,
        "top_positive": categorized["positivo"][:3],
        "top_negative": categorized["negativo"][:3]
    }


# if __name__ == "__main__":
#     ticker = "VALE3.SA"
#     resultado = enhanced_sentiment_analysis(ticker)
#     print(resultado["resume_sentiment"])
#     print("Top Positivas:")
#     for p in resultado["top_positive"]:
#         print("-", p)
#     print("Top Negativas:")
#     for n in resultado["top_negative"]:
#         print("-", n)
