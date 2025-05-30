import requests
from bs4 import BeautifulSoup
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import pipeline
import re

def preprocess_text(text):
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^A-Za-z0-9À-ÿ.,;!?() ]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def enhanced_sentiment_analysis(ticker):
    try:
        # Fontes confiáveis para notícias em português (exemplo: G1 e Exame)
        news_sources = [
            f"https://g1.globo.com/busca/?q={ticker}",
            f"https://exame.com/?s={ticker}"
        ]
        
        headlines = []
        max_headlines = 50
        
        for url in news_sources:
            try:
                r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=7)
                r.raise_for_status()
                soup = BeautifulSoup(r.text, 'html.parser')
                
                # Buscar títulos de notícias nos elementos mais prováveis em cada site
                # Exemplo G1: títulos estão em <a class="feed-post-link"
                # Exame: títulos em <h2 class="title"
                # Ajuste manual conforme a fonte
                
                if "g1.globo.com" in url:
                    elems = soup.find_all('a', class_='feed-post-link')
                elif "exame.com" in url:
                    elems = soup.find_all('h2', class_='title')
                else:
                    elems = soup.find_all(['h1','h2','h3','h4','p'])
                
                for elem in elems:
                    text = preprocess_text(elem.get_text())
                    if len(text) > 30 and text not in headlines:
                        headlines.append(text)
                    if len(headlines) >= max_headlines:
                        break
                
                if len(headlines) >= max_headlines:
                    break
            
            except Exception as e:
                print(f"Erro ao acessar {url}: {e}")
                continue
        
        if not headlines:
            return {
                "final_sentiment": "neutro",
                "message": "Nenhuma notícia encontrada para o ticker informado.",
                "resume_sentiment": "Não foi possível encontrar notícias relevantes para realizar a análise."
            }
        
        # Análise VADER (adaptada para português)
        vader = SentimentIntensityAnalyzer()
        vader_scores = [vader.polarity_scores(h)['compound'] for h in headlines]
        vader_avg = sum(vader_scores) / len(vader_scores)
        
        # Análise transformer com modelo multilíngue (mais robusto para pt)
        try:
            sentiment_pipeline = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")
            results = sentiment_pipeline(headlines[:20])
            
            # Modelo retorna rótulos 1 a 5 estrelas; vamos converter:
            # 1-2 estrelas -> negativo (-score)
            # 3 estrelas -> neutro (0)
            # 4-5 estrelas -> positivo (+score)
            transformer_scores = []
            for res in results:
                label = res['label']
                stars = int(label.split()[0])
                score = res['score']
                if stars <= 2:
                    transformer_scores.append(-score)
                elif stars == 3:
                    transformer_scores.append(0)
                else:
                    transformer_scores.append(score)
            transformer_avg = sum(transformer_scores) / len(transformer_scores) if transformer_scores else 0
        
        except Exception as e:
            print(f"Erro ao carregar transformer: {e}. Usando só VADER.")
            transformer_avg = 0
        
        # Média ponderada dos scores (pode ajustar pesos)
        final_score = (vader_avg * 0.6) + (transformer_avg * 0.4)
        
        pos_pct = len([s for s in vader_scores if s > 0.05]) / len(vader_scores) * 100
        neg_pct = len([s for s in vader_scores if s < -0.05]) / len(vader_scores) * 100
        neu_pct = 100 - pos_pct - neg_pct
        
        if final_score > 0.15:
            final_sentiment = "fortemente positivo"
        elif final_score > 0.05:
            final_sentiment = "positivo"
        elif final_score < -0.15:
            final_sentiment = "fortemente negativo"
        elif final_score < -0.05:
            final_sentiment = "negativo"
        else:
            final_sentiment = "neutro"
        
        # Manchetes mais positivas e negativas para mostrar no resumo
        top_positive = sorted(zip(headlines, vader_scores), key=lambda x: x[1], reverse=True)[:3]
        top_negative = sorted(zip(headlines, vader_scores), key=lambda x: x[1])[:3]
        
        resumo = (
            f"A análise das notícias para '{ticker}' mostrou um sentimento {final_sentiment}. "
            f"Foram analisadas {len(headlines)} manchetes. "
            f"A proporção das manchetes positivas é de {pos_pct:.2f}%, negativas {neg_pct:.2f}%, "
            f"e neutras {neu_pct:.2f}%. "
            "As manchetes mais positivas foram:\n"
        )
        for h, s in top_positive:
            resumo += f"  - \"{h}\" (score: {s:.2f})\n"
        resumo += "As manchetes mais negativas foram:\n"
        for h, s in top_negative:
            resumo += f"  - \"{h}\" (score: {s:.2f})\n"
        
        if final_sentiment in ["fortemente positivo", "positivo"]:
            resumo += "O mercado aparenta ter uma visão favorável sobre este ativo."
        elif final_sentiment in ["fortemente negativo", "negativo"]:
            resumo += "O sentimento do mercado está desfavorável para este ativo."
        else:
            resumo += "O sentimento do mercado está neutro, indicando ausência de tendência clara."
        
        resumo += "\nNota: esta análise é baseada em manchetes de notícias e pode não refletir completamente o sentimento real do mercado."
        
        return {
            "final_sentiment": final_sentiment,
            "final_score": round(final_score, 4),
            "vader_score": round(vader_avg, 4),
            "transformer_score": round(transformer_avg, 4),
            "positive_pct": round(pos_pct, 2),
            "negative_pct": round(neg_pct, 2),
            "neutral_pct": round(neu_pct, 2),
            "num_headlines": len(headlines),
            "resume_sentiment": resumo
        }
    
    except Exception as e:
        print(f"Erro na análise de sentimento: {e}")
        return {
            "final_sentiment": "neutro",
            "error": str(e),
            "resume_sentiment": "Não foi possível realizar a análise devido a um erro interno."
        }
