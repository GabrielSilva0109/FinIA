#!/usr/bin/env python
"""
Script para corrigir nomes das colunas no arquivo ml_models_advanced.py
"""

import re

def fix_columns():
    with open('ml_models_advanced.py', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Substituições necessárias
    substitutions = [
        ("data['Close']", "data['close']"),
        ("data['Volume']", "data['volume']"),
        ("data['High']", "data['high']"),
        ("data['Low']", "data['low']"),
        ('data["Close"]', 'data["close"]'),
        ('data["Volume"]', 'data["volume"]'),
        ('data["High"]', 'data["high"]'),
        ('data["Low"]', 'data["low"]'),
    ]
    
    for old, new in substitutions:
        content = content.replace(old, new)
    
    with open('ml_models_advanced.py', 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("✅ Colunas corrigidas com sucesso!")

if __name__ == "__main__":
    fix_columns()