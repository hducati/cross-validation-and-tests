import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.naive_bayes import GaussianNB


def main():
    data = pd.read_csv('census.csv')
    previsores = data.iloc[:, 0:14].values
    classe = data.iloc[:, 14].values
    prev_list = [1, 3, 5, 6, 7, 8, 9, 13]
    
    prev_encoder = LabelEncoder()

    for x in prev_list:
        previsores[:, x] = prev_encoder.fit_transform(previsores[:, x])
    scaler = StandardScaler()
    previsores = scaler.fit_transform(previsores)
    
    impute = SimpleImputer()
    impute = impute.fit(previsores)
    previsores = impute.transform(previsores)
    
    results = []
    matrix = []
    
    kfold = StratifiedKFold(10, shuffle=True, random_state=0)
    for indice_treinamento, indice_teste in kfold.split(previsores,
                                                        np.zeros((previsores.shape[0], 1))):
        classifier = GaussianNB()
        classifier.fit(previsores[indice_treinamento], classe[indice_treinamento])
        predict = classifier.predict(previsores[indice_teste])
        accuracy = accuracy_score(classe[indice_teste], predict)
        matrix.append(confusion_matrix(classe[indice_teste], predict))
        results.append(accuracy)
    
    f_accuracy = np.asarray(results)
    f_matrix = np.mean(matrix, axis=0)
    print(f_accuracy.mean())
    print(f_matrix)
    
if __name__ == '__main__':
    main()
    