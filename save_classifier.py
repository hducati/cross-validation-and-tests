import pandas as pd
import pickle
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier


base = pd.read_csv('credit-data.csv')
previsores = base.iloc[:, 1:4].values
classe = base.iloc[:, 4].values
    
impute = SimpleImputer()
impute = impute.fit(previsores[:, 1:4])
previsores[:, 1:4] = impute.transform(previsores[:, 1:4])
    
scaler = StandardScaler()
previsores = scaler.fit_transform(previsores)

classifierSVM = SVC(C=2.0)
classifierSVM.fit(previsores, classe)

classifierRandomForest = RandomForestClassifier(n_estimators=40, criterion='entropy')
classifierRandomForest.fit(previsores, classe)

# verbose = Visualizar mensagens
# max_iter = qt. epocas
classifierMLP = MLPClassifier(verbose=True, max_iter=1000,
                              tol=0.000010, solver='adam',
                              hidden_layer_sizes=(100), activation='relu',
                              batch_size=200, learning_rate_init=0.001)
classifierMLP.fit(previsores, classe)

# salvando classificador
pickle.dump(classifierSVM, open('svm_finalizado.sav', 'wb'))
pickle.dump(classifierRandomForest, open('random_forest_finalizado.sav', 'wb'))
pickle.dump(classifierMLP, open('neural_network.sav', 'wb'))