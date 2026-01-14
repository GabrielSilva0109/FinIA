#!/usr/bin/env python3
"""
Script de validação completa do projeto FinAI.
Executa testes, verificações de qualidade e gera relatório.
"""
import os
import sys
import subprocess
import json
from datetime import datetime
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def run_command(command, description):
    """Executa um comando e retorna resultado."""
    logger.info(f"Executando: {description}")
    try:
        result = subprocess.run(
            command, 
            shell=True, 
            capture_output=True, 
            text=True, 
            timeout=300
        )
        return {
            "success": result.returncode == 0,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "returncode": result.returncode
        }
    except subprocess.TimeoutExpired:
        logger.error(f"Timeout ao executar: {description}")
        return {"success": False, "error": "Timeout"}
    except Exception as e:
        logger.error(f"Erro ao executar {description}: {e}")
        return {"success": False, "error": str(e)}


def check_dependencies():
    """Verifica se as dependências estão instaladas."""
    logger.info("Verificando dependências...")
    
    dependencies = [
        "fastapi", "uvicorn", "yfinance", "pandas", "numpy", 
        "scikit-learn", "transformers", "requests", "beautifulsoup4"
    ]
    
    missing = []
    for dep in dependencies:
        result = run_command(f"pip show {dep}", f"Verificando {dep}")
        if not result["success"]:
            missing.append(dep)
    
    return missing


def run_syntax_check():
    """Verifica sintaxe dos arquivos Python."""
    logger.info("Verificando sintaxe dos arquivos...")
    
    python_files = [
        "main.py", "logic.py", "technical_indicators.py", 
        "ml_models.py", "sentiment_analysis.py", "logic_crypto.py",
        "models.py", "config.py"
    ]
    
    syntax_errors = []
    for file in python_files:
        if os.path.exists(file):
            result = run_command(f"python -m py_compile {file}", f"Sintaxe de {file}")
            if not result["success"]:
                syntax_errors.append({
                    "file": file,
                    "error": result["stderr"]
                })
    
    return syntax_errors


def run_tests():
    """Executa os testes unitários."""
    logger.info("Executando testes unitários...")
    
    test_files = ["test_main.py", "test_improved.py"]
    test_results = {}
    
    for test_file in test_files:
        if os.path.exists(test_file):
            result = run_command(f"python -m pytest {test_file} -v", f"Testes em {test_file}")
            test_results[test_file] = result
    
    return test_results


def check_api_startup():
    """Verifica se a API consegue inicializar."""
    logger.info("Testando inicialização da API...")
    
    # Tentar importar o módulo principal
    try:
        import main
        logger.info("✅ Módulo main.py importado com sucesso")
        
        # Verificar se a aplicação FastAPI foi criada
        if hasattr(main, 'app'):
            logger.info("✅ Aplicação FastAPI criada com sucesso")
            return True
        else:
            logger.error("❌ Aplicação FastAPI não encontrada")
            return False
            
    except Exception as e:
        logger.error(f"❌ Erro ao importar main.py: {e}")
        return False


def check_file_structure():
    """Verifica se a estrutura de arquivos está correta."""
    logger.info("Verificando estrutura de arquivos...")
    
    required_files = [
        "main.py", "requirements.txt", "README.md", "Dockerfile",
        ".gitignore", "config.py", "models.py"
    ]
    
    recommended_files = [
        "logic.py", "technical_indicators.py", "ml_models.py",
        "sentiment_analysis.py", "logic_crypto.py"
    ]
    
    missing_required = []
    missing_recommended = []
    
    for file in required_files:
        if not os.path.exists(file):
            missing_required.append(file)
    
    for file in recommended_files:
        if not os.path.exists(file):
            missing_recommended.append(file)
    
    return {
        "missing_required": missing_required,
        "missing_recommended": missing_recommended
    }


def generate_report(results):
    """Gera relatório de validação."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    report = f"""
# 📊 Relatório de Validação - FinAI
**Data:** {timestamp}

## ✅ Resumo Executivo
"""
    
    # Calcular score geral
    total_checks = 5
    passed_checks = 0
    
    # 1. Dependências
    missing_deps = results["dependencies"]
    if not missing_deps:
        passed_checks += 1
        report += "\n- ✅ Todas as dependências instaladas"
    else:
        report += f"\n- ❌ Dependências faltando: {', '.join(missing_deps)}"
    
    # 2. Sintaxe
    syntax_errors = results["syntax"]
    if not syntax_errors:
        passed_checks += 1
        report += "\n- ✅ Sintaxe correta em todos os arquivos"
    else:
        report += f"\n- ❌ Erros de sintaxe em {len(syntax_errors)} arquivos"
    
    # 3. Estrutura de arquivos
    file_structure = results["file_structure"]
    if not file_structure["missing_required"]:
        passed_checks += 1
        report += "\n- ✅ Estrutura de arquivos obrigatórios completa"
    else:
        report += f"\n- ❌ Arquivos obrigatórios faltando: {', '.join(file_structure['missing_required'])}"
    
    # 4. API
    if results["api_startup"]:
        passed_checks += 1
        report += "\n- ✅ API inicializa corretamente"
    else:
        report += "\n- ❌ Problemas na inicialização da API"
    
    # 5. Testes
    test_results = results["tests"]
    tests_passed = any(result["success"] for result in test_results.values())
    if tests_passed:
        passed_checks += 1
        report += "\n- ✅ Pelo menos alguns testes passaram"
    else:
        report += "\n- ❌ Nenhum teste passou com sucesso"
    
    # Score final
    score = (passed_checks / total_checks) * 100
    report += f"\n\n**Score de Qualidade: {score:.1f}%** ({passed_checks}/{total_checks} checks aprovados)\n"
    
    # Detalhes
    report += "\n## 📋 Detalhes\n"
    
    if missing_deps:
        report += f"\n### 📦 Dependências Faltando\n"
        for dep in missing_deps:
            report += f"- {dep}\n"
        report += f"\n**Solução:** `pip install {' '.join(missing_deps)}`\n"
    
    if syntax_errors:
        report += f"\n### 🐛 Erros de Sintaxe\n"
        for error in syntax_errors:
            report += f"- **{error['file']}**: {error['error']}\n"
    
    if file_structure["missing_required"]:
        report += f"\n### 📁 Arquivos Obrigatórios Faltando\n"
        for file in file_structure["missing_required"]:
            report += f"- {file}\n"
    
    if file_structure["missing_recommended"]:
        report += f"\n### 📄 Arquivos Recomendados Faltando\n"
        for file in file_structure["missing_recommended"]:
            report += f"- {file}\n"
    
    # Testes detalhados
    report += f"\n### 🧪 Resultados dos Testes\n"
    for test_file, result in test_results.items():
        status = "✅ PASSOU" if result["success"] else "❌ FALHOU"
        report += f"- **{test_file}**: {status}\n"
        
        if not result["success"] and "stderr" in result:
            report += f"  - Erro: {result['stderr'][:200]}...\n"
    
    # Recomendações
    report += f"\n## 🚀 Próximos Passos\n"
    
    if score >= 80:
        report += "✅ **Projeto em excelente estado!**\n\n"
        report += "- Considere adicionar mais testes unitários\n"
        report += "- Documente as APIs com exemplos\n"
        report += "- Configure CI/CD para deploy automático\n"
    elif score >= 60:
        report += "⚠️ **Projeto funcional, mas precisa de melhorias:**\n\n"
        report += "- Corrigir os problemas identificados acima\n"
        report += "- Adicionar testes para maior cobertura\n"
        report += "- Revisar documentação\n"
    else:
        report += "🔧 **Projeto precisa de atenção:**\n\n"
        report += "- Priorizar correção de erros críticos\n"
        report += "- Instalar dependências faltantes\n"
        report += "- Executar testes básicos\n"
    
    return report


def main():
    """Função principal de validação."""
    logger.info("🚀 Iniciando validação do projeto FinAI...")
    
    # Mudar para o diretório do projeto
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    results = {}
    
    # 1. Verificar dependências
    results["dependencies"] = check_dependencies()
    
    # 2. Verificar sintaxe
    results["syntax"] = run_syntax_check()
    
    # 3. Verificar estrutura de arquivos
    results["file_structure"] = check_file_structure()
    
    # 4. Verificar inicialização da API
    results["api_startup"] = check_api_startup()
    
    # 5. Executar testes
    results["tests"] = run_tests()
    
    # 6. Gerar relatório
    report = generate_report(results)
    
    # Salvar relatório
    with open("validation_report.md", "w", encoding="utf-8") as f:
        f.write(report)
    
    # Mostrar no console
    print(report)
    
    logger.info("✅ Validação concluída! Relatório salvo em 'validation_report.md'")
    
    # Salvar resultados em JSON para processamento posterior
    with open("validation_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    return results


if __name__ == "__main__":
    main()