# Atividade P3 - Previsão da nota final

**Integrantes:**
- Endrigo Gustavo Brandão de Oliveira
- Nickolas Maia de Araujo

Criamos uma base fictícia com 300 alunos e treinamos uma regressão linear para prever a nota final a partir de horas de estudo por semana, frequência (%), atividades entregues (de 10) e nota anterior. Depois calculamos média, desvio padrão e intervalo de erro das previsões.

## Como rodar

```
pip install -r requirements.txt
python previsao_notas.py
```

O script gera tudo de novo (a seed é fixa, então os números saem iguais):

- `dados/alunos.csv` – base fake
- `graficos/` – 6 gráficos (distribuição das notas, correlação, real x previsto, erros etc.)
- `resultados/estatisticas.txt` – médias, desvios, intervalo de erro e métricas
- `resultados/previsoes_teste.csv` – nota real, prevista e erro de cada aluno do teste

## Resultado em resumo

| Métrica | Valor |
|---|---|
| Média do erro | -0,11 |
| Desvio padrão do erro | 0,69 |
| Intervalo de 95% dos erros | -1,47 a 1,24 |
| MAE | 0,54 |
| RMSE | 0,69 |
| R² (teste) | 0,61 |

Em média o modelo erra cerca de meio ponto e 86,7% das previsões ficaram a no máximo 1 ponto da nota real. A análise completa está no relatório em Word (`Relatorio_P3.docx`).
