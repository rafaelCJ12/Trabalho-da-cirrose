import pandas as pd

from pathlib import Path

from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from sklearn.ensemble import RandomForestClassifier

from imblearn.over_sampling import SMOTE

import xgboost as xgb


def tratamentoColunaTexto(base, nomeColuna):
    base[nomeColuna] = base[nomeColuna].fillna('Não Informado')
    base[nomeColuna] = base[nomeColuna].str.upper().str.strip()
    base = pd.get_dummies(base, columns=[nomeColuna], drop_first=True, dtype=int)


    return base

        

def tratamentoBase(b):
    #excluindo a coluna ID
    b.drop(columns=['ID'], inplace=True)
    b.drop(columns=['N_Days'], inplace=True)

    b['Staus_binario'] = b['Status'].apply(lambda x: 1 if x=='D' else 0)

    b.drop(columns=['Status'], inplace=True)
        
    mapeamento = {'M': 0, 'F': 1}
    b['Sex'] = b['Sex'].map(mapeamento)
    
    b = tratamentoColunaTexto(b, 'Ascites')
    b = tratamentoColunaTexto(b, 'Hepatomegaly')
    b = tratamentoColunaTexto(b, 'Spiders')
    b = tratamentoColunaTexto(b, 'Edema')
    b = tratamentoColunaTexto(b, 'Drug')
    
    
    
    return b

def trataColuna(base, nomeColuna, valor, tipo):
    base[nomeColuna] = base[nomeColuna].fillna(valor)
    base[nomeColuna] = base[nomeColuna].astype(tipo)
    return base



def classificacaoKnn(X_treino, y_treino, X_teste, y_teste, tipoBase):
    knn = KNeighborsClassifier(n_neighbors=5)

    knn.fit(X_treino, y_treino)

    previsao = knn.predict(X_teste)

    print("\n--- MATRIZ DE CONFUSÃO CLASSIFICAO KNN COM BASE " + tipoBase + " ---")
    print(confusion_matrix(y_teste, previsao))

    print("\n--- RELATÓRIO DE DESEMPENHO CLASSIFICACAO KNN COM BASE "+ tipoBase + " ---")
    print(classification_report(y_teste, previsao))

    return

def classificacaoRandomForest(X_treino, y_treino, X_teste, y_teste, tipoBase):
    rf = RandomForestClassifier(random_state=42, n_estimators=100)
    rf.fit(X_treino, y_treino)

    previsao = rf.predict(X_teste)

    print("\n--- MATRIZ DE CONFUSÃO CLASSIFICACAO RANDOM FOREST COM BASE " + tipoBase + " ---")
    print(confusion_matrix(y_teste, previsao))

    print("\n--- RELATÓRIO DE DESEMPENHO CLASSIFICACAO RANDOM FOREST COM BASE " + tipoBase + " ---")
    print(classification_report(y_teste, previsao))


    return

def classificacaoXGBoost(X_treino, y_treino, X_teste, y_teste, tipoBase):
    modelo = xgb.XGBClassifier(
    n_estimators=100,       # Número de árvores que serão criadas sequencialmente
    learning_rate=0.1,      # Taxa de aprendizado (quanto menor, mais passos lentos e precisos)
    max_depth=4,            # Profundidade máxima de cada árvore
    random_state=42         # Garante que os resultados sejam reproduzíveis
    )

    modelo.fit(X_treino, y_treino)

    previsao = modelo.predict(X_teste)

    print("\n--- MATRIZ DE CONFUSÃO CLASSIFICACAO XGBOOST COM BASE " + tipoBase + " ---")
    print(confusion_matrix(y_teste, previsao))

    print("\n--- RELATÓRIO DE DESEMPENHO CLASSIFICACAO XGBOOST COM BASE " + tipoBase + " ---")
    print(classification_report(y_teste, previsao))


    return




def main():
    base = pd.read_csv(Path(__file__).resolve().parent / "cirrhosis.csv", sep=',')
    
    base = tratamentoBase(base)

    #y representa a coluna Status, que eh o que queremos classificar
    y = base['Staus_binario']

    #retiro da base de dados a coluna Status para iniciar a classificacao
    X = base.drop('Staus_binario', axis='columns')
    print(y.head())

    base.to_csv(Path(__file__).resolve().parent / "cirrhosis_tratada.csv", index=False)
    
    #seperacao do conjunto de treinamento e de teste
    X_treino, X_teste, y_treino, y_teste = train_test_split(X, y, test_size=0.30, random_state=42)

    #aqui trato valores nulos usando a mediana de cada coluna (numerica) do conjunto de treinamento
    colunasTipos = {'Age': int, 'Bilirubin': float, 'Cholesterol': int, 'Albumin': float, 'Copper': int,
                    'Alk_Phos': float, 'SGOT': float, 'Tryglicerides': int, 'Platelets': int, 'Prothrombin': float,
                    'Stage': int}

    for coluna, tipo in colunasTipos.items():
        if coluna in X_treino.columns:
            valor = X_treino[coluna].median()
            X_treino = trataColuna(X_treino, coluna, valor, tipo)
            X_teste = trataColuna(X_teste, coluna, valor, tipo)
    
    #fim do tratamento

    #usado para deixar os dados de cada coluna na mesma escala
    scaler = StandardScaler()

    #aplica o escalador no conjunto de treino
    X_treino_escalado = scaler.fit_transform(X_treino)

    X_teste_escalado = scaler.transform(X_teste)
    
    classificacaoKnn(X_treino_escalado, y_treino, X_teste_escalado, y_teste, "DESBALANCEADA")

    classificacaoRandomForest(X_treino_escalado, y_treino, X_teste_escalado, y_teste, "DESBALANCEADA")

    classificacaoXGBoost(X_treino_escalado, y_treino, X_teste_escalado, y_teste, "DESBALANCEADA")

    smote = SMOTE(random_state=42)

    #o conjunto final de treino ja escalado, agora serah balanceado
    X_treino_final, y_treino_final = smote.fit_resample(X_treino_escalado, y_treino)

    classificacaoKnn(X_treino_final, y_treino_final, X_teste_escalado, y_teste, "BALANCEADA")

    classificacaoRandomForest(X_treino_final, y_treino_final, X_teste_escalado, y_teste, "BALANCEADA")

    classificacaoXGBoost(X_treino_final, y_treino_final, X_teste_escalado, y_teste, "BALANCEADA")
    
main()
