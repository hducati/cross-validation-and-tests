import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


def main():
    base = pd.read_csv('files/credit-data.csv')
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

    t_results = []
    for x in range(30):
        results = []
        kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=x)
        for indice_treinamento, indice_teste in kfold.split(previsores,
                                                            np.zeros((previsores.shape[0], 1))):
            # classificador = GaussianNB()
            classificador = DecisionTreeClassifier()
            classificador.fit(previsores[indice_treinamento], classe[indice_treinamento])
            previsoes = classificador.predict(previsores[indice_teste])
            precisao = accuracy_score(classe[indice_teste], previsoes)
            results.append(precisao)
        results = np.asarray(results)
        mean = results.mean()
        t_results.append(mean)
    t_results = np.asarray(t_results)

    for x in range(t_results.size):
        print(str(t_results[x]).replace('.', ','))
    

if __name__ == '__main__':
    main()