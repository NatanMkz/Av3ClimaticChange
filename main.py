import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import bartlett
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm

def cria_grafico(df):
    media_temp = df.groupby(['Year'], as_index=False)['Temp'].mean()
    plt.figure(figsize=(10, 6))
    sns.regplot(x='Year', y='Temp', data=media_temp, scatter=True, line_kws={"color": "red"}, ci=None)  # Sem intervalo de confiança
    plt.title('Tendência da Temperatura ao Longo dos Anos com Linha de Regressão')
    plt.xlabel('Ano')
    plt.ylabel('Temperatura Média')
    plt.grid(True)
    plt.show()
    # Calculando a correlação entre as variáveis
    
def plot_grafic_correla(df):
    scaler = MinMaxScaler()
    media_gases = df.groupby(['Year'], as_index=False)[['CO2', 'CH4', 'N2O','Temp']].mean()
    media_gases[['CO2', 'CH4', 'N2O', 'Temp']] = scaler.fit_transform(
    media_gases[['CO2', 'CH4', 'N2O', 'Temp']]
    )
    
    
    plt.figure(figsize=(10, 6))

    # Linha para CO2
    sns.lineplot(data=media_gases, x='Year', y='CO2', label='CO2')

    # Linha para CH4
    sns.lineplot(data=media_gases, x='Year', y='CH4', label='CH4')

    # Linha para N2O
    sns.lineplot(data=media_gases, x='Year', y='N2O', label='N2O')

    # Linha para a temperatura
    sns.lineplot(data=media_gases, x='Year', y='Temp', label='Temp')

    # Adicionar título e rótulos
    plt.title('Mudança de Gases e Temperatura ao Longo dos Anos')
    plt.xlabel('Ano')
    plt.ylabel('Valores Médios')
    plt.legend()  # Exibe a legenda para identificar cada linha
    plt.grid(True)  # Adiciona uma grade para facilitar a leitura

    # Mostrar o gráfico
    plt.show()
    correlations = media_gases.corr()
    print("Correlação entre variáveis:")
    print(correlations,'\n')

def homogeneidade(df):
    variances = {}
    # Testando homogeneidade para cada uma das variáveis por ano

    for column in ['CO2', 'CH4', 'N2O', 'Temp']:
        # Agrupar por ano e coletar dados para o teste de Bartlett
        grouped = [group[column].dropna() for _, group in df.groupby('Year')]
        # Aplicar o teste de Bartlett
        stat, p_value = bartlett(*grouped)
        # Guardar os resultados
        variances[column] = {'statistic': stat, 'p_value': p_value}

    # Imprimir resultados
    print("Teste de Bartlett para homogeneidade das variâncias:")
    for column, result in variances.items():
        print(f"{column}: Estatística = {result['statistic']:.4f}, p-valor = {result['p_value']:.4f}")

def grafico_qq(df):
    # Crie uma matriz de subplot
    # s de 1 linha por 3 colunas
    fig, axs = plt.subplots(1, 4, figsize=(12, 4))  # 1 linha, 3 colunas

    # Colunas para as quais queremos criar gráficos Q-Q
    columns = ['CO2', 'CH4', 'N2O','Temp']

    # Criar gráficos Q-Q para cada coluna
    for i, column in enumerate(columns):
        sm.qqplot(df[column].dropna(), line='s', ax=axs[i])
        axs[i].set_title(f"Gráfico Q-Q para {column}")

    # Ajustar layout para evitar sobreposições
    plt.tight_layout()

    # Mostrar os gráficos
    plt.show()

def anova(df):
    modelo_anova = ols('Temp ~ CO2 + CH4 + N2O', data=df).fit()

    # Aplicar ANOVA para verificar a significância do modelo
    anova_results = anova_lm(modelo_anova)

    # Exibir resultados da ANOVA
    print(anova_results)


df = pd.read_excel("AV03_Modelos_Lineares_Climatic_Change_ver00.xlsx",sheet_name="Climatic_Change")



cria_grafico(df)
plot_grafic_correla(df)
grafico_qq(df)
homogeneidade(df)

anova(df)
