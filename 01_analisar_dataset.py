import os
from collections import Counter

import matplotlib.pyplot as plt

dataset_path = "imagens/utkcropped"

idades = []
generos = []
etnias = []

for filename in os.listdir(dataset_path):
    if not filename.endswith(".jpg"):
        continue

    try:
        idade, genero, etnia, _ = filename.split("_")

        idades.append(int(idade))
        generos.append(int(genero))
        etnias.append(int(etnia))

    except Exception as e:
        print(f"Erro em {filename}: {e}")

# Contagem
genero_count = Counter(generos)
etnia_count = Counter(etnias)

# Faixas de idade
faixas = {"0-20": 0, "21-40": 0, "41-60": 0, "60+": 0}

for idade in idades:
    if idade <= 20:
        faixas["0-20"] += 1
    elif idade <= 40:
        faixas["21-40"] += 1
    elif idade <= 60:
        faixas["41-60"] += 1
    else:
        faixas["60+"] += 1

print("\n=== DISTRIBUIÇÃO DE GÊNERO ===")
print("0 = Homem | 1 = Mulher")
print(genero_count)

print("\n=== DISTRIBUIÇÃO DE ETNIA ===")
print("0=Branco, 1=Negro, 2=Asiático, 3=Indiano, 4=Outros")
print(etnia_count)

print("\n=== DISTRIBUIÇÃO DE IDADE ===")
for faixa, count in faixas.items():
    print(f"{faixa}: {count}")

# Plotar gráficos
fig, axs = plt.subplots(1, 3, figsize=(18, 5))

# Gráfico de Gênero
axs[0].bar(["Homem", "Mulher"], [genero_count[0], genero_count[1]])
axs[0].set_title("Distribuição de Gênero")
axs[0].set_ylabel("Quantidade")

# Gráfico de Etnia
etnia_labels = ["Branco", "Negro", "Asiático", "Indiano", "Outros"]
etnia_values = [etnia_count.get(i, 0) for i in range(5)]
axs[1].bar(etnia_labels, etnia_values)
axs[1].set_title("Distribuição de Etnia")
axs[1].set_ylabel("Quantidade")
axs[1].tick_params(axis="x", rotation=20)

# Gráfico de Faixas de Idade
faixa_labels = list(faixas.keys())
faixa_values = list(faixas.values())
axs[2].bar(faixa_labels, faixa_values)
axs[2].set_title("Distribuição de Idade")
axs[2].set_ylabel("Quantidade")

plt.tight_layout()

# Salvar gráficos individualmente
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)

# Gênero
fig_genero, ax_genero = plt.subplots()
ax_genero.bar(["Homem", "Mulher"], [genero_count[0], genero_count[1]])
ax_genero.set_title("Distribuição de Gênero")
ax_genero.set_ylabel("Quantidade")
fig_genero.tight_layout()
fig_genero.savefig(os.path.join(output_dir, "distribuicao_genero.png"))
plt.close(fig_genero)

# Etnia
fig_etnia, ax_etnia = plt.subplots()
ax_etnia.bar(etnia_labels, etnia_values)
ax_etnia.set_title("Distribuição de Etnia")
ax_etnia.set_ylabel("Quantidade")
ax_etnia.tick_params(axis="x", rotation=20)
fig_etnia.tight_layout()
fig_etnia.savefig(os.path.join(output_dir, "distribuicao_etnia.png"))
plt.close(fig_etnia)

# Idade
fig_idade, ax_idade = plt.subplots()
ax_idade.bar(faixa_labels, faixa_values)
ax_idade.set_title("Distribuição de Idade")
ax_idade.set_ylabel("Quantidade")
fig_idade.tight_layout()
fig_idade.savefig(os.path.join(output_dir, "distribuicao_idade.png"))
plt.close(fig_idade)

# Não exibe os gráficos, apenas salva
