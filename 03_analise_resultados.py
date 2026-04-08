import csv
import os
from collections import Counter, defaultdict
from statistics import mean

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

ARQUIVO_RESULTADO = "saida_modelo/resultado_final.csv"
OUTPUT_DIR = "saida_modelo/analises"

MAPA_GENERO_REAL = {0: "Homem", 1: "Mulher"}
MAPA_ETNIA_REAL = {0: "Branco", 1: "Negro", 2: "Asiático", 3: "Indiano", 4: "Outros"}

MAPA_GENERO_PRED = {"Man": "Homem", "Woman": "Mulher"}
MAPA_ETNIA_PRED = {
    "white": "Branco",
    "black": "Negro",
    "asian": "Asiático",
    "indian": "Indiano",
    "middle eastern": "Oriente Médio",
    "latino hispanic": "Latino/Hispânico",
}

ORDEM_GENERO = ["Homem", "Mulher"]
ORDEM_ETNIA = ["Branco", "Negro", "Asiático", "Indiano", "Outros"]
ORDEM_FAIXA = ["0-20", "21-40", "41-60", "60+"]


def faixa_etaria(idade: int) -> str:
    if idade <= 20:
        return "0-20"
    elif idade <= 40:
        return "21-40"
    elif idade <= 60:
        return "41-60"
    return "60+"


def safe_mean(valores):
    return mean(valores) if valores else 0.0


def ensure_output_dir():
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def salvar_csv(caminho, cabecalho, linhas):
    with open(caminho, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(cabecalho)
        writer.writerows(linhas)


def ordenar_categorias_presentes(categorias, ordem_base):
    presentes = list(dict.fromkeys(categorias))
    ordenadas = [c for c in ordem_base if c in presentes]
    extras = [c for c in presentes if c not in ordem_base]
    return ordenadas + extras


def carregar_registros():
    registros = []

    with open(ARQUIVO_RESULTADO, mode="r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            try:
                idade_real = int(row["idade_real"])
                genero_real = int(row["genero_real"])
                etnia_real = int(row["etnia_real"])
                idade_pred = int(float(row["idade_pred"]))
                genero_pred = row["genero_pred"].strip()
                etnia_pred = row["etnia_pred"].strip()

                confianca = (
                    float(row["confianca"])
                    if "confianca" in row and row["confianca"]
                    else None
                )

                genero_real_nome = MAPA_GENERO_REAL.get(
                    genero_real, f"Desconhecido({genero_real})"
                )
                etnia_real_nome = MAPA_ETNIA_REAL.get(
                    etnia_real, f"Desconhecido({etnia_real})"
                )
                genero_pred_nome = MAPA_GENERO_PRED.get(genero_pred, genero_pred)
                etnia_pred_nome = MAPA_ETNIA_PRED.get(etnia_pred, etnia_pred)

                erro_idade_signed = idade_pred - idade_real
                erro_idade_abs = abs(erro_idade_signed)

                registro = {
                    "arquivo": row["arquivo"],
                    "idade_real": idade_real,
                    "genero_real": genero_real_nome,
                    "etnia_real": etnia_real_nome,
                    "idade_pred": idade_pred,
                    "genero_pred": genero_pred_nome,
                    "etnia_pred": etnia_pred_nome,
                    "confianca": confianca,
                    "erro_idade_signed": erro_idade_signed,
                    "erro_idade": erro_idade_abs,
                    "faixa_etaria": faixa_etaria(idade_real),
                    "acerto_genero": genero_real_nome == genero_pred_nome,
                    "acerto_etnia": etnia_real_nome == etnia_pred_nome,
                }

                registros.append(registro)

            except Exception as e:
                print(f"Erro ao processar linha {row}: {e}")

    return registros


def imprimir_resumo_geral(registros):
    total = len(registros)
    confs = [r["confianca"] for r in registros if r["confianca"] is not None]
    erros_abs = [r["erro_idade"] for r in registros]
    erros_signed = [r["erro_idade_signed"] for r in registros]

    print("\n" + "=" * 70)
    print("RESUMO GERAL")
    print("=" * 70)
    print(f"Total de imagens analisadas: {total}")
    print(f"Erro médio absoluto de idade: {safe_mean(erros_abs):.2f}")
    print(
        f"Erro médio assinado de idade (predito - real): {safe_mean(erros_signed):.2f}"
    )
    print(
        f"Acurácia geral de gênero: {100 * safe_mean([1 if r['acerto_genero'] else 0 for r in registros]):.2f}%"
    )
    print(
        f"Acurácia geral de etnia: {100 * safe_mean([1 if r['acerto_etnia'] else 0 for r in registros]):.2f}%"
    )

    if confs:
        print(f"Confiança média: {safe_mean(confs):.4f}")
    else:
        print("Confiança média: não disponível")

    print(
        f"Proporção de superestimação de idade (>0): {100 * sum(e > 0 for e in erros_signed) / total:.2f}%"
    )
    print(
        f"Proporção de subestimação de idade (<0): {100 * sum(e < 0 for e in erros_signed) / total:.2f}%"
    )

    salvar_csv(
        os.path.join(OUTPUT_DIR, "resumo_geral.csv"),
        [
            "total_imagens",
            "erro_medio_absoluto_idade",
            "erro_medio_assinado_idade",
            "acuracia_genero",
            "acuracia_etnia",
            "confianca_media",
        ],
        [
            [
                total,
                round(safe_mean(erros_abs), 4),
                round(safe_mean(erros_signed), 4),
                round(
                    100
                    * safe_mean([1 if r["acerto_genero"] else 0 for r in registros]),
                    4,
                ),
                round(
                    100 * safe_mean([1 if r["acerto_etnia"] else 0 for r in registros]),
                    4,
                ),
                round(safe_mean(confs), 4) if confs else "",
            ]
        ],
    )


def salvar_distribuicoes(registros):
    cont_genero_real = Counter(r["genero_real"] for r in registros)
    cont_etnia_real = Counter(r["etnia_real"] for r in registros)
    cont_etnia_pred = Counter(r["etnia_pred"] for r in registros)
    cont_faixa = Counter(r["faixa_etaria"] for r in registros)

    print("\nDistribuição de gênero real:")
    for k, v in cont_genero_real.items():
        print(f"  {k}: {v}")

    print("\nDistribuição de etnia real:")
    for k, v in cont_etnia_real.items():
        print(f"  {k}: {v}")

    print("\nDistribuição de etnia prevista:")
    for k, v in cont_etnia_pred.items():
        print(f"  {k}: {v}")

    print("\nDistribuição por faixa etária:")
    for k, v in cont_faixa.items():
        print(f"  {k}: {v}")

    salvar_csv(
        os.path.join(OUTPUT_DIR, "distribuicoes.csv"),
        ["tipo", "categoria", "quantidade"],
        (
            [["genero_real", k, v] for k, v in cont_genero_real.items()]
            + [["etnia_real", k, v] for k, v in cont_etnia_real.items()]
            + [["etnia_pred", k, v] for k, v in cont_etnia_pred.items()]
            + [["faixa_etaria", k, v] for k, v in cont_faixa.items()]
        ),
    )


def analisar_por_grupo(
    registros, chave_grupo, titulo, arquivo_saida_csv, ordem_preferida=None
):
    grupos = defaultdict(list)
    for r in registros:
        grupos[r[chave_grupo]].append(r)

    categorias = list(grupos.keys())
    if ordem_preferida:
        categorias = ordenar_categorias_presentes(categorias, ordem_preferida)

    print("\n" + "=" * 70)
    print(f"ANÁLISE POR {titulo.upper()}")
    print("=" * 70)

    linhas_csv = []
    erros = []
    erros_signed = []
    confiancas = []
    acc_genero = []
    acc_etnia = []
    quantidades = []

    for grupo in categorias:
        itens = grupos[grupo]
        qtd = len(itens)
        erro_medio = safe_mean([r["erro_idade"] for r in itens])
        erro_signed_medio = safe_mean([r["erro_idade_signed"] for r in itens])

        confs = [r["confianca"] for r in itens if r["confianca"] is not None]
        confianca_media = safe_mean(confs) if confs else None

        acuracia_genero = 100 * safe_mean(
            [1 if r["acerto_genero"] else 0 for r in itens]
        )
        acuracia_etnia = 100 * safe_mean([1 if r["acerto_etnia"] else 0 for r in itens])

        print(f"\n{grupo}")
        print(f"  Quantidade: {qtd}")
        print(f"  Erro médio absoluto de idade: {erro_medio:.2f}")
        print(f"  Erro médio assinado de idade: {erro_signed_medio:.2f}")
        print(f"  Acurácia de gênero: {acuracia_genero:.2f}%")
        print(f"  Acurácia de etnia: {acuracia_etnia:.2f}%")
        if confianca_media is not None:
            print(f"  Confiança média: {confianca_media:.4f}")
        else:
            print("  Confiança média: não disponível")

        linhas_csv.append(
            [
                grupo,
                qtd,
                round(erro_medio, 4),
                round(erro_signed_medio, 4),
                round(acuracia_genero, 4),
                round(acuracia_etnia, 4),
                round(confianca_media, 4) if confianca_media is not None else "",
            ]
        )

        quantidades.append(qtd)
        erros.append(erro_medio)
        erros_signed.append(erro_signed_medio)
        confiancas.append(confianca_media if confianca_media is not None else 0)
        acc_genero.append(acuracia_genero)
        acc_etnia.append(acuracia_etnia)

    salvar_csv(
        os.path.join(OUTPUT_DIR, arquivo_saida_csv),
        [
            "grupo",
            "quantidade",
            "erro_medio_absoluto_idade",
            "erro_medio_assinado_idade",
            "acuracia_genero",
            "acuracia_etnia",
            "confianca_media",
        ],
        linhas_csv,
    )

    # quantidade
    plt.figure(figsize=(8, 4))
    plt.bar(categorias, quantidades)
    plt.title(f"Quantidade por {titulo}")
    plt.ylabel("Quantidade")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"quantidade_por_{chave_grupo}.png"))
    plt.close()

    # erro absoluto
    plt.figure(figsize=(8, 4))
    plt.bar(categorias, erros)
    plt.title(f"Erro Médio Absoluto de Idade por {titulo}")
    plt.ylabel("Erro Médio")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"erro_idade_por_{chave_grupo}.png"))
    plt.close()

    # erro assinado
    plt.figure(figsize=(8, 4))
    plt.bar(categorias, erros_signed)
    plt.axhline(0, linestyle="--")
    plt.title(f"Erro Médio Assinado de Idade por {titulo}")
    plt.ylabel("Erro Médio (Predito - Real)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"erro_signed_idade_por_{chave_grupo}.png"))
    plt.close()

    # acurácia de gênero
    plt.figure(figsize=(8, 4))
    plt.bar(categorias, acc_genero)
    plt.title(f"Acurácia de Gênero por {titulo}")
    plt.ylabel("Acurácia (%)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"acuracia_genero_por_{chave_grupo}.png"))
    plt.close()

    # acurácia de etnia
    plt.figure(figsize=(8, 4))
    plt.bar(categorias, acc_etnia)
    plt.title(f"Acurácia de Etnia por {titulo}")
    plt.ylabel("Acurácia (%)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"acuracia_etnia_por_{chave_grupo}.png"))
    plt.close()

    # confiança
    if any(c > 0 for c in confiancas):
        plt.figure(figsize=(8, 4))
        plt.bar(categorias, confiancas)
        plt.title(f"Confiança Média por {titulo}")
        plt.ylabel("Confiança Média")
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f"confianca_por_{chave_grupo}.png"))
        plt.close()


def gerar_graficos_gerais(registros):
    erros_abs = [r["erro_idade"] for r in registros]
    erros_signed = [r["erro_idade_signed"] for r in registros]

    # Histograma erro absoluto idade
    plt.figure(figsize=(8, 4))
    sns.histplot(erros_abs, bins=30, kde=True)
    plt.title("Histograma dos Erros Absolutos de Idade")
    plt.xlabel("Erro Absoluto de Idade")
    plt.ylabel("Frequência")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "histograma_erro_idade.png"))
    plt.close()

    # Histograma erro assinado idade
    plt.figure(figsize=(8, 4))
    sns.histplot(erros_signed, bins=30, kde=True)
    plt.title("Distribuição do Erro Assinado de Idade")
    plt.xlabel("Erro de Idade (Predito - Real)")
    plt.ylabel("Frequência")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "histograma_erro_idade_signed.png"))
    plt.close()

    # Boxplot por faixa etária
    faixas = ordenar_categorias_presentes(
        [r["faixa_etaria"] for r in registros], ORDEM_FAIXA
    )
    df = pd.DataFrame(registros)

    plt.figure(figsize=(8, 4))
    sns.boxplot(data=df, x="faixa_etaria", y="erro_idade", order=faixas)
    plt.title("Boxplot do Erro de Idade por Faixa Etária")
    plt.xlabel("Faixa Etária")
    plt.ylabel("Erro Absoluto de Idade")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "boxplot_erro_idade_por_faixa.png"))
    plt.close()

    # Boxplot por etnia
    etnias = ordenar_categorias_presentes(
        [r["etnia_real"] for r in registros], ORDEM_ETNIA
    )
    plt.figure(figsize=(9, 4))
    sns.boxplot(data=df, x="etnia_real", y="erro_idade", order=etnias)
    plt.title("Boxplot do Erro de Idade por Etnia Real")
    plt.xlabel("Etnia Real")
    plt.ylabel("Erro Absoluto de Idade")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "boxplot_erro_idade_por_etnia.png"))
    plt.close()


def gerar_matriz_confusao_etnia(registros):
    etnias_reais = [r["etnia_real"] for r in registros]
    etnias_pred = [r["etnia_pred"] for r in registros]

    conf_matrix = pd.crosstab(
        pd.Series(etnias_reais, name="Real"),
        pd.Series(etnias_pred, name="Prevista"),
    )

    print("\n" + "=" * 70)
    print("MATRIZ DE CONFUSÃO DE ETNIA")
    print("=" * 70)
    print(conf_matrix)

    conf_matrix.to_csv(
        os.path.join(OUTPUT_DIR, "matriz_confusao_etnia.csv"), encoding="utf-8"
    )

    plt.figure(figsize=(9, 6))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
    plt.title("Matriz de Confusão de Etnia (Real x Prevista)")
    plt.ylabel("Etnia Real")
    plt.xlabel("Etnia Prevista")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "matriz_confusao_etnia.png"))
    plt.close()


def salvar_casos_grotescos(registros):
    print("\n" + "=" * 70)
    print("CASOS MAIS GRITANTES")
    print("=" * 70)

    maiores_erros_idade = sorted(
        registros, key=lambda r: r["erro_idade"], reverse=True
    )[:20]
    erros_genero = [r for r in registros if not r["acerto_genero"]][:20]
    erros_etnia = [r for r in registros if not r["acerto_etnia"]][:20]

    print("\nTop 10 maiores erros de idade:")
    for r in maiores_erros_idade[:10]:
        print(
            f"{r['arquivo']} | idade real: {r['idade_real']} | "
            f"idade predita: {r['idade_pred']} | erro abs: {r['erro_idade']} | "
            f"erro signed: {r['erro_idade_signed']} | "
            f"gênero real: {r['genero_real']} | etnia real: {r['etnia_real']}"
        )

    print("\nTop 10 erros de gênero:")
    for r in erros_genero[:10]:
        print(
            f"{r['arquivo']} | gênero real: {r['genero_real']} | "
            f"gênero predito: {r['genero_pred']} | "
            f"idade real: {r['idade_real']} | etnia real: {r['etnia_real']}"
        )

    print("\nTop 10 erros de etnia:")
    for r in erros_etnia[:10]:
        print(
            f"{r['arquivo']} | etnia real: {r['etnia_real']} | "
            f"etnia predita: {r['etnia_pred']} | "
            f"idade real: {r['idade_real']} | gênero real: {r['genero_real']}"
        )

    salvar_csv(
        os.path.join(OUTPUT_DIR, "maiores_erros_idade.csv"),
        [
            "arquivo",
            "idade_real",
            "idade_pred",
            "erro_idade_abs",
            "erro_idade_signed",
            "genero_real",
            "genero_pred",
            "etnia_real",
            "etnia_pred",
        ],
        [
            [
                r["arquivo"],
                r["idade_real"],
                r["idade_pred"],
                r["erro_idade"],
                r["erro_idade_signed"],
                r["genero_real"],
                r["genero_pred"],
                r["etnia_real"],
                r["etnia_pred"],
            ]
            for r in maiores_erros_idade
        ],
    )

    salvar_csv(
        os.path.join(OUTPUT_DIR, "erros_genero.csv"),
        [
            "arquivo",
            "genero_real",
            "genero_pred",
            "idade_real",
            "idade_pred",
            "etnia_real",
            "etnia_pred",
        ],
        [
            [
                r["arquivo"],
                r["genero_real"],
                r["genero_pred"],
                r["idade_real"],
                r["idade_pred"],
                r["etnia_real"],
                r["etnia_pred"],
            ]
            for r in erros_genero
        ],
    )

    salvar_csv(
        os.path.join(OUTPUT_DIR, "erros_etnia.csv"),
        [
            "arquivo",
            "etnia_real",
            "etnia_pred",
            "idade_real",
            "idade_pred",
            "genero_real",
            "genero_pred",
        ],
        [
            [
                r["arquivo"],
                r["etnia_real"],
                r["etnia_pred"],
                r["idade_real"],
                r["idade_pred"],
                r["genero_real"],
                r["genero_pred"],
            ]
            for r in erros_etnia
        ],
    )


def salvar_resumo_textual():
    texto = """Arquivos gerados nesta análise:

1. resumo_geral.csv
2. distribuicoes.csv
3. resumo_por_genero.csv
4. resumo_por_etnia.csv
5. resumo_por_faixa.csv
6. matriz_confusao_etnia.csv
7. maiores_erros_idade.csv
8. erros_genero.csv
9. erros_etnia.csv

Gráficos principais recomendados para apresentação:
- erro_idade_por_etnia_real.png
- erro_idade_por_faixa_etaria.png
- acuracia_genero_por_etnia_real.png
- acuracia_etnia_por_etnia_real.png
- matriz_confusao_etnia.png
- histograma_erro_idade.png
- histograma_erro_idade_signed.png
- boxplot_erro_idade_por_faixa.png
"""
    with open(
        os.path.join(OUTPUT_DIR, "README_ANALISE.txt"), "w", encoding="utf-8"
    ) as f:
        f.write(texto)


def main():
    ensure_output_dir()
    registros = carregar_registros()

    if not registros:
        print("Nenhum registro válido encontrado.")
        return

    imprimir_resumo_geral(registros)
    salvar_distribuicoes(registros)
    gerar_graficos_gerais(registros)
    gerar_matriz_confusao_etnia(registros)

    analisar_por_grupo(
        registros,
        chave_grupo="genero_real",
        titulo="Gênero Real",
        arquivo_saida_csv="resumo_por_genero.csv",
        ordem_preferida=ORDEM_GENERO,
    )

    analisar_por_grupo(
        registros,
        chave_grupo="etnia_real",
        titulo="Etnia Real",
        arquivo_saida_csv="resumo_por_etnia.csv",
        ordem_preferida=ORDEM_ETNIA,
    )

    analisar_por_grupo(
        registros,
        chave_grupo="faixa_etaria",
        titulo="Faixa Etária",
        arquivo_saida_csv="resumo_por_faixa.csv",
        ordem_preferida=ORDEM_FAIXA,
    )

    salvar_casos_grotescos(registros)
    salvar_resumo_textual()

    print("\n" + "=" * 70)
    print("ANÁLISE FINALIZADA")
    print("=" * 70)
    print(f"Arquivos salvos em: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
