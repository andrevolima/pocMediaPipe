"""Ponto de entrada — API por padrão; terminal para testes manuais.

Uso:
    python main.py
    python main.py --manual
"""

from pathlib import Path

import argparse


def main():
    parser = argparse.ArgumentParser(description="API postural ou testes manuais no terminal")
    parser.add_argument("--manual", action="store_true")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    if not args.manual:
        import uvicorn
        uvicorn.run("src.api:app", host=args.host, port=args.port)
        return
    from src.services.analiseService import processar_pasta
    pasta_base = Path(__file__).resolve().parent
    processar_pasta(
        pasta_entrada=pasta_base / "entrada",
        pasta_saida=pasta_base / "saida",
    )


if __name__ == "__main__":
    main()
