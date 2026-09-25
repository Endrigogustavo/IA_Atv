import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

SEED = 42
QTD_ALUNOS = 300

os.makedirs("dados", exist_ok=True)
os.makedirs("graficos", exist_ok=True)
os.makedirs("resultados", exist_ok=True)


def gerar_base(n, seed):
    rng = np.random.default_rng(seed)

    horas_estudo = np.clip(rng.normal(7, 3.5, n), 0, 20).round(1)
    frequencia = np.clip(rng.normal(82, 11, n), 40, 100).round(0)
    base_ativ = 3 + (frequencia - 40) / 60 * 6 + rng.normal(0, 1.5, n)
    atividades = np.clip(base_ativ, 0, 10).round(0).astype(int)

    nota_anterior = np.clip(rng.normal(6.2, 1.8, n), 0, 10).round(1)

    nota_final = (
        -0.3
        + 0.18 * horas_estudo
        + 0.025 * frequencia
        + 0.20 * atividades
        + 0.35 * nota_anterior
        + rng.normal(0, 0.7, n)
    )
    nota_final = np.clip(nota_final, 0, 10).round(1)

    return pd.DataFrame({
        "aluno_id": np.arange(1, n + 1),
        "horas_estudo": horas_estudo,
        "frequencia": frequencia,
        "atividades_entregues": atividades,
        "nota_anterior": nota_anterior,
        "nota_final": nota_final,
    })


df = gerar_base(QTD_ALUNOS, SEED)
df.to_csv("dados/alunos.csv", index=False)

print("Primeiras linhas da base:")
print(df.head(), "\n")

colunas = ["horas_estudo", "frequencia", "atividades_entregues", "nota_anterior", "nota_final"]
desc = df[colunas].agg(["mean", "std", "min", "max"]).T
desc.columns = ["media", "desvio_padrao", "minimo", "maximo"]
print("Estatistica descritiva da base:")
print(desc.round(2), "\n")

X = df[["horas_estudo", "frequencia", "atividades_entregues", "nota_anterior"]]
y = df["nota_final"]

X_treino, X_teste, y_treino, y_teste = train_test_split(
    X, y, test_size=0.2, random_state=SEED
)

modelo = LinearRegression()
modelo.fit(X_treino, y_treino)

previsto = modelo.predict(X_teste)
previsto = np.clip(previsto, 0, 10)

print("Coeficientes do modelo:")
for nome, coef in zip(X.columns, modelo.coef_):
    print(f"  {nome:22s} {coef:.4f}")
print(f"  {'intercepto':22s} {modelo.intercept_:.4f}\n")

erros = y_teste.values - previsto
erros_abs = np.abs(erros)

media_erro = erros.mean()
dp_erro = erros.std(ddof=1)
n_teste = len(erros)

mae = mean_absolute_error(y_teste, previsto)
rmse = np.sqrt(mean_squared_error(y_teste, previsto))
r2 = r2_score(y_teste, previsto)

lim_inf = media_erro - 1.96 * dp_erro
lim_sup = media_erro + 1.96 * dp_erro
dentro = np.mean((erros >= lim_inf) & (erros <= lim_sup)) * 100

t_crit = stats.t.ppf(0.975, df=n_teste - 1)
ic_inf = media_erro - t_crit * dp_erro / np.sqrt(n_teste)
ic_sup = media_erro + t_crit * dp_erro / np.sqrt(n_teste)

ate_meio = np.mean(erros_abs <= 0.5) * 100
ate_um = np.mean(erros_abs <= 1.0) * 100

cv_r2 = cross_val_score(LinearRegression(), X, y, cv=5, scoring="r2")

baseline = np.full_like(y_teste, y_treino.mean(), dtype=float)
mae_baseline = mean_absolute_error(y_teste, baseline)

resumo = f"""RESULTADOS - ATIVIDADE P3
Integrantes: Endrigo Gustavo Brandao de Oliveira e Nickolas Maia de Araujo

Base: {QTD_ALUNOS} alunos | treino: {len(X_treino)} | teste: {n_teste}

Notas reais (teste)      media = {y_teste.mean():.2f}   desvio = {y_teste.std(ddof=1):.2f}
Notas previstas (teste)  media = {previsto.mean():.2f}   desvio = {previsto.std(ddof=1):.2f}

Erro (real - previsto)
  media ............................ {media_erro:.3f}
  desvio padrao .................... {dp_erro:.3f}
  intervalo de 95% dos erros ....... [{lim_inf:.2f} ; {lim_sup:.2f}]  ({dentro:.1f}% dos alunos de teste ficaram dentro)
  IC 95% da media do erro .......... [{ic_inf:.3f} ; {ic_sup:.3f}]
  erro absoluto maximo ............. {erros_abs.max():.2f}

Metricas
  MAE  (erro medio absoluto) ....... {mae:.3f}
  RMSE ............................. {rmse:.3f}
  R2 (teste) ....................... {r2:.3f}
  R2 validacao cruzada (5 folds) ... {cv_r2.mean():.3f} +- {cv_r2.std():.3f}
  MAE chutando sempre a media ...... {mae_baseline:.3f}

Previsoes com erro de ate 0,5 ponto: {ate_meio:.1f}%
Previsoes com erro de ate 1,0 ponto: {ate_um:.1f}%

Coeficientes
""" + "\n".join(f"  {n:22s} {c:.4f}" for n, c in zip(X.columns, modelo.coef_)) +\
    f"\n  {'intercepto':22s} {modelo.intercept_:.4f}\n"

print(resumo)

with open("resultados/estatisticas.txt", "w", encoding="utf-8") as f:
    f.write(resumo)
    f.write("\nEstatistica descritiva da base\n")
    f.write(desc.round(2).to_string())

pd.DataFrame({
    "nota_real": y_teste.values,
    "nota_prevista": previsto.round(2),
    "erro": erros.round(2),
}).to_csv("resultados/previsoes_teste.csv", index=False)

plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": 0.3})

fig, ax = plt.subplots(figsize=(7, 4.2))
ax.hist(df["nota_final"], bins=20, color="#4C72B0", edgecolor="white")
ax.axvline(df["nota_final"].mean(), color="red", ls="--",
           label=f"média = {df['nota_final'].mean():.2f}")
ax.set_title("Distribuição da nota final (base completa)")
ax.set_xlabel("Nota final")
ax.set_ylabel("Quantidade de alunos")
ax.legend()
fig.tight_layout()
fig.savefig("graficos/01_distribuicao_notas.png", dpi=150)
plt.close(fig)

corr = df[colunas].corr()
rotulos = ["Horas estudo", "Frequência", "Atividades", "Nota anterior", "Nota final"]
fig, ax = plt.subplots(figsize=(6.5, 5.2))
im = ax.imshow(corr, cmap="Blues", vmin=-1, vmax=1)
ax.set_xticks(range(len(rotulos)))
ax.set_yticks(range(len(rotulos)))
ax.set_xticklabels(rotulos, rotation=35, ha="right")
ax.set_yticklabels(rotulos)
ax.grid(False)
for i in range(len(rotulos)):
    for j in range(len(rotulos)):
        v = corr.iloc[i, j]
        ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                color="white" if v > 0.6 else "black")
fig.colorbar(im, ax=ax, fraction=0.046)
ax.set_title("Correlação entre as variáveis")
fig.tight_layout()
fig.savefig("graficos/02_correlacao.png", dpi=150)
plt.close(fig)

fig, ax = plt.subplots(figsize=(6, 5.5))
ax.scatter(y_teste, previsto, alpha=0.7, color="#4C72B0", edgecolor="k", lw=0.3)
ax.plot([0, 10], [0, 10], "r--", label="previsão perfeita")
ax.fill_between([0, 10], [0 - 1, 10 - 1], [0 + 1, 10 + 1], color="orange",
                alpha=0.15, label="faixa de ±1 ponto")
ax.set_xlim(2, 10)
ax.set_ylim(2, 10)
ax.set_xlabel("Nota real")
ax.set_ylabel("Nota prevista")
ax.set_title(f"Nota real x prevista (R2 = {r2:.2f})")
ax.legend(loc="upper left")
fig.tight_layout()
fig.savefig("graficos/03_real_vs_previsto.png", dpi=150)
plt.close(fig)

fig, ax = plt.subplots(figsize=(7, 4.2))
ax.hist(erros, bins=15, color="#55A868", edgecolor="white")
ax.axvline(media_erro, color="red", ls="--", label=f"média = {media_erro:.2f}")
ax.axvline(lim_inf, color="gray", ls=":", label=f"intervalo 95% [{lim_inf:.2f} ; {lim_sup:.2f}]")
ax.axvline(lim_sup, color="gray", ls=":")
ax.set_title("Distribuição dos erros (real - previsto)")
ax.set_xlabel("Erro em pontos")
ax.set_ylabel("Quantidade de alunos")
ax.legend()
fig.tight_layout()
fig.savefig("graficos/04_distribuicao_erros.png", dpi=150)
plt.close(fig)

fig, ax = plt.subplots(figsize=(7, 4.2))
ax.scatter(previsto, erros, alpha=0.7, color="#C44E52", edgecolor="k", lw=0.3)
ax.axhline(0, color="black", lw=1)
ax.axhline(lim_inf, color="gray", ls=":")
ax.axhline(lim_sup, color="gray", ls=":")
ax.set_title("Erro x nota prevista")
ax.set_xlabel("Nota prevista")
ax.set_ylabel("Erro (real - previsto)")
fig.tight_layout()
fig.savefig("graficos/05_erro_vs_previsto.png", dpi=150)
plt.close(fig)

impacto = modelo.coef_ * X.std().values
fig, ax = plt.subplots(figsize=(7, 3.8))
ax.barh(["Horas estudo", "Frequência", "Atividades", "Nota anterior"], impacto, color="#8172B2")
ax.set_title("Impacto de cada variável na nota (coef. x desvio padrão)")
ax.set_xlabel("Variacao na nota prevista (pontos)")
fig.tight_layout()
fig.savefig("graficos/06_impacto_variaveis.png", dpi=150)
plt.close(fig)

print("Graficos salvos na pasta graficos/")
