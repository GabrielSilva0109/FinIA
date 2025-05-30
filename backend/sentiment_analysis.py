import requests
from bs4 import BeautifulSoup
from transformers import pipeline
import re
import hashlib
from collections import defaultdict
import time # Para adicionar delays entre as requisições
import yfinance as yf
import re

def preprocess_text(text):
    """Remove URLs, caracteres não alfanuméricos e espaços extras."""
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^A-Za-z0-9À-ÿ.,;!?() ]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# # Função auxiliar para obter o nome da empresa a partir do ticker
def get_company_name_from_ticker(ticker: str) -> str:
    ticker = ticker.upper().strip()

    # Verifica se é ticker brasileiro (ex: VALE3, PETR4, ITUB4 etc.)
    is_brazilian = bool(re.match(r'^[A-Z]{4}[0-9]$', ticker)) or ticker.endswith('.SA')

    # Se for brasileiro e não tiver .SA, adiciona
    if is_brazilian and not ticker.endswith('.SA'):
        ticker += ".SA"

    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        return info.get("longName") or info.get("shortName") or ticker
    except Exception as e:
        print(f"Erro ao buscar nome da empresa para o ticker '{ticker}': {e}")
        return ticker

def fetch_news_from_source(url, parser, tag, attr_key, attr_value, relevant_terms):
    """
    Busca manchetes de uma URL, filtrando pela presença dos termos relevantes.
    Agora também tenta extrair o resumo/corpo da notícia, se possível, para análise mais profunda.
    """
    headlines_and_content = []
    try:
        r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=15)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, parser)

        for tag_item in soup.find_all(tag, {attr_key: attr_value}):
            headline_text = preprocess_text(tag_item.get_text())

            # Apenas adicione a manchete se ela for relevante e tiver um comprimento razoável
            if any(term.lower() in headline_text.lower() for term in relevant_terms) and len(headline_text) > 30:
                # Tenta encontrar um resumo ou primeiro parágrafo próximo à manchete
                # Esta parte é altamente dependente da estrutura HTML de cada site.
                # Será preciso inspecionar cada fonte para ajustar estes seletores.
                article_content = ""
                parent = tag_item.find_parent(["div", "article"]) # Tenta subir no DOM
                if parent:
                    # Exemplo: buscar um <p> com classe de resumo ou um div com conteúdo
                    summary_elem = parent.find("p", class_=re.compile("summary|description|resumo")) # Adapte padrões
                    if not summary_elem: # Tenta outra heurística, por exemplo, o primeiro parágrafo
                        summary_elem = parent.find("div", class_=re.compile("body|content")) # Adapte padrões
                        if summary_elem:
                            first_p = summary_elem.find("p")
                            if first_p:
                                summary_elem = first_p

                    if summary_elem:
                        article_content = preprocess_text(summary_elem.get_text())

                # Combina manchete e conteúdo para análise, se o conteúdo for substancial
                text_for_analysis = headline_text
                if article_content and len(article_content) > 50: # Se o resumo for decente
                    text_for_analysis = f"{headline_text}. {article_content}"

                headlines_and_content.append({"text": text_for_analysis, "source_url": url})

    except requests.exceptions.RequestException as e:
        print(f"Erro de requisição em {url}: {e}")
    except Exception as e:
        print(f"Erro ao buscar em {url}: {e}")
    return headlines_and_content

def deduplicate_headlines_by_hash(headlines_data):
    """Filtra duplicatas exatas de manchetes usando hashing."""
    unique_headlines = []
    seen_hashes = set()
    for item in headlines_data:
        headline_hash = hashlib.md5(item['text'].encode('utf-8')).hexdigest()
        if headline_hash not in seen_hashes:
            unique_headlines.append(item)
            seen_hashes.add(headline_hash)
    return unique_headlines

def filter_relevance_and_financial_context(headlines_data, primary_term):
    """
    Filtra notícias para garantir que são relevantes e com contexto financeiro.
    Prioriza a presença de termos financeiros e desconsidera temas irrelevantes.
    """
    filtered = []
    primary_term_lower = primary_term.lower()
    
    # Termos financeiros chave que indicam relevância para o mercado
    finance_keywords = [
        "lucro", "receita", "guidance", "investimento", "ação", "dividendos",
        "mercado", "balanço", "prejuízo", "crescimento", "queda", "valor",
        "negócios", "expansão", "aquisição", "fusão", "patrimônio"
    ]
    
    # Termos que indicam irrelevância (não financeiro, ex: esportes, política geral)
    exclude_keywords = [
        "polícia", "crime", "celebridade", "futebol", "entretenimento",
        "eleição", "candidato", "partido", "jogos olímpicos", "novela"
    ]

    for item in headlines_data:
        text_lower = item['text'].lower()
        
        # Garante que o termo principal da empresa esteja presente
        if primary_term_lower not in text_lower:
            continue

        # Verifica a presença de termos de exclusão (se tiver, pula a notícia)
        if any(kw in text_lower for kw in exclude_keywords):
            continue

        # Prioriza notícias que mencionam termos financeiros
        if any(kw in text_lower for kw in finance_keywords):
            filtered.append(item)
        else:
            # Se não tem termos financeiros, pode incluir se for muito diretamente sobre a empresa
            # e não contiver termos de exclusão.
            # Você pode ajustar este limiar de relevância conforme o desejado.
            filtered.append(item) # Ou adicionar uma lógica de pontuação aqui

    return filtered

def enhanced_sentiment_analysis(ticker):
    """
    Realiza a análise de sentimento aprimorada para um dado ticker,
    sem usar um dicionário de sinônimos pré-definido.
    """
    # Preferencialmente use um modelo de sentimento financeiro, se disponível e em português
    # Ex: sentiment_pipeline = pipeline("sentiment-analysis", model="ProsusAI/finbert")
    # Para demonstração e compatibilidade, mantemos o nlptown
    sentiment_pipeline = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")

    # Determina o termo principal para busca (o ticker em si e o nome da empresa)
    company_name = get_company_name_from_ticker(ticker)
    # relevant_terms incluirá o ticker e o nome da empresa
    relevant_terms = [ticker.upper()]
    if company_name and company_name.lower() != ticker.lower(): # Evita duplicar se nome e ticker forem iguais
        relevant_terms.append(company_name)
    
    print(f"Termos de busca para '{ticker}': {relevant_terms}")

    # Lista de fontes de notícias (ajuste os seletores de tag/attr_key/attr_value conforme o site)
    sources = [
        # (URL formatada, parser, tag da manchete, atributo da manchete, valor do atributo)
        # Lembre-se: o '{query}' será preenchido com o `ticker` ou `company_name`
        (f"https://valor.globo.com/busca/?q={company_name}", "html.parser", "a", "class", "feed-post-link"),
        (f"https://www.infomoney.com.br/?s={company_name}", "html.parser", "a", "class", "hl-title"),
        (f"https://exame.com/?s={company_name}", "html.parser", "a", "class", "title"),
        (f"https://br.investing.com/search/?q={company_name}", "html.parser", "a", "class", "title"),
        (f"https://busca.uol.com.br/?q={company_name}", "html.parser", "a", "class", "thumb-caption"),
        (f"https://g1.globo.com/busca/?q={company_name}", "html.parser", "a", "class", "widget--info__text-container"),
        (f"https://busca.estadao.com.br/?q={company_name}", "html.parser", "a", "class", "resultado"),
        # Yahoo Finance (pode exigir ajustes específicos se o lookup não funcionar bem)
        (f"https://br.financas.yahoo.com/lookup?s={ticker}", "html.parser", "a", "class", "Fz(16px)"),
        (f"https://www.seudinheiro.com/?s={company_name}", "html.parser", "a", "class", "post-title"),
        (f"https://www.suno.com.br/noticias/?s={company_name}", "html.parser", "a", "class", "entry-title"),
        (f"https://www.moneytimes.com.br/?s={company_name}", "html.parser", "h2", "class", "card-title"),
        (f"https://www.investnews.com.br/?s={company_name}", "html.parser", "h3", "class", "noticia-title"),
    ]

    all_headlines_data = []

    for url, parser, tag, attr_key, attr_value in sources:
        # Usamos o nome da empresa na URL de busca para cobrir mais resultados
        # A filtragem por `relevant_terms` na função `fetch_news_from_source` garante a relevância
        print(f"Buscando em: {url}")
        headlines = fetch_news_from_source(url, parser, tag, attr_key, attr_value, relevant_terms)
        print(f"  Encontradas {len(headlines)} manchetes/itens.")
        all_headlines_data.extend(headlines)
        time.sleep(1) # Pequeno delay para evitar bloqueios

    print(f"\nTotal de itens coletados (bruto): {len(all_headlines_data)}")

    # 1. Filtrar duplicatas exatas primeiro
    unique_headlines = deduplicate_headlines_by_hash(all_headlines_data)
    print(f"Total de manchetes únicas (após hashing): {len(unique_headlines)}")

    # 2. Filtrar por relevância e contexto financeiro
    # Aqui, passamos o `company_name` como o termo principal para o filtro.
    final_headlines_for_analysis = filter_relevance_and_financial_context(unique_headlines, company_name)
    print(f"Total de manchetes únicas e relevantes para análise: {len(final_headlines_for_analysis)}")

    if not final_headlines_for_analysis:
        return {
            "final_sentiment": "neutro",
            "transformer_score": 0.0,
            "headlines_analyzed": 0,
            "resume_sentiment": "Nenhuma notícia relevante encontrada.",
            "top_positive": [],
            "top_negative": []
        }

    # Limita o número de manchetes para análise para evitar sobrecarga (pode ajustar)
    headlines_to_analyze_texts = [item['text'] for item in final_headlines_for_analysis[:50]]
    
    results = sentiment_pipeline(headlines_to_analyze_texts)

    transformer_scores = []
    categorized = defaultdict(list)

    for i, res in enumerate(results):
        label = res['label']
        stars = int(label.split()[0])
        score = res['score']
        headline_text = headlines_to_analyze_texts[i] # Pega o texto original

        if stars <= 2:
            sentiment_score = -score
            categorized["negativo"].append(headline_text)
        elif stars == 3:
            sentiment_score = 0
            categorized["neutro"].append(headline_text)
        else:
            sentiment_score = score
            categorized["positivo"].append(headline_text)
        
        transformer_scores.append(sentiment_score)

    avg_score = sum(transformer_scores) / len(transformer_scores) if transformer_scores else 0

    if avg_score > 0.30: # Ajustar limiares para "fortemente"
        final_sentiment = "fortemente positivo"
    elif avg_score > 0.10:
        final_sentiment = "positivo"
    elif avg_score < -0.30:
        final_sentiment = "fortemente negativo"
    elif avg_score < -0.10:
        final_sentiment = "negativo"
    else:
        final_sentiment = "neutro"

    summary = (
        f"Analisando {len(final_headlines_for_analysis)} manchetes únicas e relevantes de diversas fontes sobre '{ticker}', "
        f"o sentimento geral é **{final_sentiment}**.\n\n"
        f"- Positivas: {len(categorized['positivo'])}\n"
        f"- Neutras: {len(categorized['neutro'])}\n"
        f"- Negativas: {len(categorized['negativo'])}\n"
    )

    return {
        "final_sentiment": final_sentiment,
        "transformer_score": round(avg_score, 4),
        "headlines_analyzed": len(final_headlines_for_analysis),
        "resume_sentiment": summary,
        "top_positive": categorized["positivo"][:3],
        "top_negative": categorized["negativo"][:3]
    }
