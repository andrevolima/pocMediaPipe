"""Ponto de entrada — processa imagens de ciclistas da pasta 'entrada'.

Uso:
    python main.py
"""

from pathlib import Path

from src.services.analiseService import processar_pasta


def main():
    pasta_base = Path(__file__).resolve().parent
    processar_pasta(
        pasta_entrada=pasta_base / "entrada",
        pasta_saida=pasta_base / "saida",
    )


if __name__ == "__main__":
    main()