import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score


def main():
    base = pd.read_csv('credit-data.csv')
    previsores = base.iloc[:, 1:4].values
    classe = base.iloc[:, 4].values
    
    impute = SimpleImputer()
    impute = impute.fit(previsores[:, 1:4])
    previsores[:, 1:4] = impute.transform(previsores[:, 1:4])
    
    scaler = StandardScaler()
    previsores = scaler.fit_transform(previsores)
    
    a = np.zeros(5)
    previsores.shape
    b = np.zeros((previsores.shape[0], 1))
    
    kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)
    results = []
    matriz = []
    for indice_treinamento, indice_teste in kfold.split(previsores,
                                                        np.zeros((previsores.shape[0], 1))):
        classificador = GaussianNB()
        classificador.fit(previsores[indice_treinamento], classe[indice_treinamento])
        previsoes = classificador.predict(previsores[indice_teste])
        precisao = accuracy_score(classe[indice_teste], previsoes)
        matriz.append(confusion_matrix(classe[indice_teste], previsoes))
        results.append(precisao)

# transformando em um array    
results = np.asarray(results)
# obtendo o resultado final, no caso, a média
results.mean()
# recebendo a média das matrizes de confusao.
# axis=0(especifica que vai pegar a linha, caso nao seja espeficicado, ele não vai conseguir
# realizar a média.)
matriz_final = np.mean(matriz, axis=0)


if __name__ == '__main__':
    main()
