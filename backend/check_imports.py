#!/usr/bin/env python3
"""
Script rápido para verificar se todos os arquivos podem ser importados.
"""
import sys
import importlib

def test_imports():
    """Testa se todos os módulos podem ser importados."""
    modules = [
        'config',
        'models', 
        'technical_indicators',
        'ml_models',
        'sentiment_analysis',
        'logic_crypto',
        'logic',
        'main'
    ]
    
    print("🔍 Testando imports...")
    errors = []
    
    for module in modules:
        try:
            importlib.import_module(module)
            print(f"✅ {module}")
        except Exception as e:
            print(f"❌ {module}: {e}")
            errors.append(f"{module}: {e}")
    
    if errors:
        print(f"\n❌ {len(errors)} erros encontrados:")
        for error in errors:
            print(f"  - {error}")
        return False
    else:
        print(f"\n✅ Todos os {len(modules)} módulos importados com sucesso!")
        return True

if __name__ == "__main__":
    success = test_imports()
    sys.exit(0 if success else 1)