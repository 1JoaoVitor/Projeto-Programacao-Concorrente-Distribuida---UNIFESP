import numpy as np

np.random.seed(42)

N = 50000000

K = 30
CENTROIDES_REAIS = np.linspace(start=10, stop=500, num=K)

DESVIO_PADRAO = 5 

pontos_por_cluster = N // K
dados = []
for i in range(K):
    cluster = np.random.normal(loc=CENTROIDES_REAIS[i], scale=DESVIO_PADRAO, size=pontos_por_cluster)
    dados.extend(cluster)

np.random.shuffle(dados)

np.savetxt("dados.csv", dados, fmt='%.4f')

# Gerar centroides iniciais (um pouco deslocados dos reais para o algoritmo trabalhar)
centroides_iniciais = [c + np.random.uniform(-5, 5) for c in CENTROIDES_REAIS]

np.savetxt("centroides_iniciais.csv", centroides_iniciais, fmt='%.4f')

print(f"Arquivo 'dados.csv' com {len(dados)} pontos gerado.")
print(f"Arquivo 'centroides_iniciais.csv' com {len(centroides_iniciais)} centroides gerado.")