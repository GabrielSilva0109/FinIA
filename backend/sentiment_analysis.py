import feedparser
from transformers import pipeline
import re
import time
import random

def preprocess_text(text):
    """
    Limpa o texto removendo URLs, caracteres especiais e espaços extras.
    """
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^A-Za-z0-9À-ÿ.,;!?() ]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def fetch_google_news_rss(query, num_results=10):
    """
    Busca notícias usando o feed RSS do Google News.
    """
    feed_url = f"https://news.google.com/rss/search?q={requests.utils.quote(query)}&hl=pt-BR&gl=BR&ceid=BR:pt-419"
    feed = feedparser.parse(feed_url)
    headlines = []
    for entry in feed.entries[:num_results]:
        title = preprocess_text(entry.title)
        summary = preprocess_text(entry.summary) if 'summary' in entry else ''
        full_text = f"{title}. {summary}"
        headlines.append(full_text)
    return headlines

def enhanced_sentiment_analysis(ticker: str):
    """
    Realiza a análise de sentimento de notícias sobre um ticker financeiro,
    utilizando o feed RSS do Google News.

    Args:
        ticker (str): O código do ativo financeiro (ex: 'PETR4').

    Returns:
        dict: Um dicionário contendo o sentimento final, pontuação,
              número de manchetes analisadas, resumo e as top notícias.
    """
    print(f"Iniciando análise de sentimento para o ticker: {ticker}")

    sentiment_pipeline = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")

    all_headlines_snippets = []

    search_queries = [
        f"notícias {ticker} bolsa hoje",
        f"ações {ticker} mercado financeiro",
        f"{ticker} resultados {ticker} balanço",
        f"{ticker} investimentos {ticker} perspectivas",
        f"{ticker} dividendos JCP",
        f"{ticker} análise técnica fundamentalista",
        f"{ticker} comunicado ao mercado",
        f"{ticker} cotação {ticker} valor",
        f"{ticker} setor {ticker} economia",
        f"{ticker} perspectivas {ticker} crescimento",
        f"{ticker} Ibovespa",
        f"{ticker} notícia negativa",
        f"{ticker} notícia positiva"
    ]

    for query_text in search_queries:
        time.sleep(random.uniform(1, 3))  # Atraso para evitar bloqueios
        snippets = fetch_google_news_rss(query_text, num_results=10)
        all_headlines_snippets.extend(snippets)

    print(f"\nTotal de {len(all_headlines_snippets)} manchetes/snippets coletados de todas as buscas.")

    if not all_headlines_snippets:
        return {
            "final_sentiment": "neutro",
            "resume_sentiment": "Nenhuma notícia relevante encontrada através do feed RSS do Google News. O sentimento geral é neutro por falta de dados.",
            "headlines_analyzed": 0,
            "transformer_score": 0.0,
            "top_positive": [],
            "top_negative": []
        }

    headlines_to_analyze = all_headlines_snippets[:200]
    print(f"Analisando {len(headlines_to_analyze)} manchetes/snippets para sentimento...")

    results = sentiment_pipeline(headlines_to_analyze)

    transformer_scores = []
    categorized_news = {"positivo": [], "neutro": [], "negativo": []}

    for i, res in enumerate(results):
        label = res['label']
        stars = int(label.split()[0])
        score = res['score']
        headline = headlines_to_analyze[i]

        if stars <= 2:
            transformer_scores.append(-score)
            categorized_news["negativo"].append({"text": headline, "score": res['score'], "label": label})
        elif stars == 3:
            transformer_scores.append(0)
            categorized_news["neutro"].append({"text": headline, "score": res['score'], "label": label})
        else:
            transformer_scores.append(score)
            categorized_news["positivo"].append({"text": headline, "score": res['score'], "label": label})

    if not transformer_scores:
        return {
            "final_sentiment": "neutro",
            "resume_sentiment": "Nenhuma manchete com sentimento analisável após o processamento. O sentimento geral é neutro.",
            "headlines_analyzed": len(headlines_to_analyze),
            "transformer_score": 0.0,
            "top_positive": [],
            "top_negative": []
        }

    avg_score = sum(transformer_scores) / len(transformer_scores)

    if avg_score > 0.30:
        final_sentiment = "fortemente positivo"
    elif avg_score > 0.10:
        final_sentiment = "positivo"
    elif avg_score < -0.30:
        final_sentiment = "fortemente negativo"
    elif avg_score < -0.10:
        final_sentiment = "negativo"
    else:
        final_sentiment = "neutro"

    categorized_news["positivo"].sort(key=lambda x: x['score'], reverse=True)
    categorized_news["negativo"].sort(key=lambda x: x['score'], reverse=True)

    summary = (
        f"Análise Completa de Sentimento para **{ticker}**:\n\n"
        f"Foram coletadas **{len(all_headlines_snippets)}** manchetes e snippets (usando o feed RSS do Google News) e analisadas **{len(headlines_to_analyze)}** delas. "
        f"A pontuação média de sentimento do modelo (variando de -1 a 1) foi de **{avg_score:.4f}**.\n\n"
        f"O sentimento geral predominante para as notícias recentes sobre **{ticker}** é **{final_sentiment.upper()}**.\n\n"
        f"Detalhes da Análise:\n"
        f"- Notícias Positivas: **{len(categorized_news['positivo'])}**\n"
        f"- Notícias Neutras: **{len(categorized_news['neutro'])}**\n"
        f"- Notícias Negativas: **{len(categorized_news['negativo'])}**\n"
    )

    return {
        "final_sentiment": final_sentiment,
        "transformer_score": round(avg_score, 4),
        "headlines_analyzed": len(headlines_to_analyze),
        "total_headlines_collected": len(all_headlines_snippets),
        "resume_sentiment": summary,
        "top_positive": [item['text'] for item in categorized_news["positivo"][:5]],
        "top_negative": [item['text'] for item in categorized_news["negativo"][:5]],
        "all_positive_count": len(categorized_news['positivo']),
        "all_negative_count": len(categorized_news['negativo']),
        "all_neutral_count": len(categorized_news['neutro']),
    }

# --- Exemplo de Uso ---
if __name__ == "__main__":
    ticker_exemplo = "VALE3"  # Mude o ticker para o que desejar

    print("="*80)
    print("ATENÇÃO: Este código utiliza o feed RSS do Google News.")
    print("É uma abordagem mais estável e menos propensa a bloqueios.")
    print("="*80)
    time.sleep(3)  # Pausa para o usuário ler o aviso

    resultado_analise = enhanced_sentiment_analysis(ticker_exemplo)

    print("\n" + "="*60)
    print("        RESULTADO FINAL DA ANÁLISE DE SENTIMENTO")
    print("="*60)
    print(resultado_analise['resume_sentiment'])

    print("-" * 60)
    print(f"Pontuação Média do Modelo: {resultado_analise['transformer_score']:.4f}")
    print(f"Total de Manchetes/Snippets Coletados: {resultado_analise['total_headlines_collected']}")
    print(f"Manchetes/Snippets Analisados: {resultado_analise['headlines_analyzed']}")
    print("-" * 60)

    if resultado_analise['top_positive']:
        print("\n>>> Principais Notícias Positivas (Top 5):")
        for i, headline in enumerate(resultado_analise['top_positive']):
            print(f"  {i+1}. {headline}")
    else:
        print("\nNão foram encontradas notícias positivas relevantes.")

    if resultado_analise['top_negative']:
        print("\n>>> Principais Notícias Negativas (Top 5):")
        for i, headline in enumerate(resultado_analise['top_negative']):
            print(f"  {i+1}. {headline}")
    else:
        print("\nNão foram encontradas notícias negativas relevantes.")

    print("\n" + "="*60)
    print("Análise Concluída.")
    print("="*60)
