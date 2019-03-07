import pandas as pd
import numpy as np
import pickle
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


base = pd.read_csv('credit-data.csv')
previsores = base.iloc[:, 1:4].values
classe = base.iloc[:, 4].values
    
impute = SimpleImputer()
impute = impute.fit(previsores[:, 1:4])
previsores[:, 1:4] = impute.transform(previsores[:, 1:4])
    
scaler = StandardScaler()
previsores = scaler.fit_transform(previsores)

svm = pickle.load(open('svm_finalizado.sav', 'rb'))
random_forest = pickle.load(open('random_forest_finalizado.sav', 'rb'))
mlp = pickle.load(open('neural_network.sav', 'rb'))

r_svm = svm.score(previsores, classe)
r_rf = random_forest.score(previsores, classe)
r_mlp = mlp.score(previsores, classe)

novo_registro = [[50000, 40, 5000]]
novo_registro = np.asarray(novo_registro)
# -1 não pega linhas
# vai trabalhar apenas com colunas
# melhor consistencia dos resultados
# é necessario que ele fique nesse formato para fazer o escalonamento
# caso nao esteja, irá zerar todos os valores
novo_registro = novo_registro.reshape(-1, 1)
novo_registro = scaler.fit_transform(novo_registro)
# voltando ao formato original
novo_registro = novo_registro.reshape(-1, 3)

r_svm = svm.predict(novo_registro)
r_rf = random_forest.predict(novo_registro)
r_mlp = mlp.predict(novo_registro)
