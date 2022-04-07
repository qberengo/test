import pandas as pd
import numpy as np
from flask import render_template,request, Flask, make_response, abort
from werkzeug.exceptions import Unauthorized,BadRequest
from sklearn.model_selection import train_test_split as tts
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

app = Flask(__name__)

#             Récupération des users et password
df_users=pd.read_csv('credentials.csv', sep =',')
users = dict([(i,j) for i,j in zip (df_users['username'],df_users['password'])])

#             Préparation des données
# On crée le data frame df à partir du fichier csv
df = pd.read_csv("bike.csv",sep=',', header=0)
# Remplacement des valeurs de météo par des catégories numériques
df = df.replace(to_replace = ['clear','cloudy', 'rainy', 'snowy'], value= [1, 2, 3,4])
# Conversion du type de colonne
df['weathersit'] = df['weathersit'].astype('int')
# Suppression des doublons
df = df.drop_duplicates()
# Définition du dictionnaire de fonctions à appliquer
function_to_apply = {
    'weathersit' : ['mean'],
    'hum' : ['min','max','mean'],
    'windspeed' : ['min','max','mean'],
    'temp' : ['min','max','mean'],
    'atemp' : ['min','max','mean'],
    'cnt' : ['sum'],
}
# Agrégation des données par jour et affichage
df_groupby=df.groupby("dteday").agg(function_to_apply).reset_index()
df_groupby.columns = df_groupby.columns.droplevel()
df_groupby.columns = ['dteday','mean_weathersit', 'min_hum', 'max_hum', 'mean_hum','min_windspeed','max_windspeed', 'mean_windspeed','min_temp','max_temp','mean_temp', 'min_atemp', 'max_atemp', 'mean_atemp', 'sum_cnt']
# Création d'un DataFrame sans la colonne heure
df_ = df_groupby.drop(columns = ['min_hum', 'max_hum', 'min_windspeed','max_windspeed','min_temp','max_temp','min_atemp', 'max_atemp'])
# Définition de la colonne dteday d'un type temps 
df_["dteday"] = df_["dteday"].astype(np.datetime64)
# On crée une nouvelle colonne à la fin avec le nombre de vélo du jour suivant
df_['cnt_']=0
for i in range(df_.shape[0]-1):
    df_.iloc[i,7]=df_.iloc[i+1,6]
# On supprime la dernière ligne
df_.drop(df_.tail(1).index,inplace=True)
# Création colonne "weekday" dans le DataFrame contenant une catégorisation du jour de la semaine avec 0 = lundi ... et 6 = dimanche
df_["weekday"] = df_["dteday"].apply(lambda x : x.weekday())
# Définition du DataFrame définitif de travail selon les features séléctionnées : 
bike = df_.drop(["mean_windspeed", "dteday", "mean_temp", "mean_hum" ], axis = 1)
# Division du DataFrame en 2 autres : X pour les variables explicatives et y pour la cible
X = bike.drop([ "cnt_"], axis = 1)
y = bike["cnt_"]
# On applique la fonction train_test_split
X_train,X_test,y_train,y_test=tts(X,y,test_size=0.30, random_state=42)    

def authenticate_user(username, password):
    authenticated_user = False
    if username in users.keys():
        if users[username] == password:
            authenticated_user = True
    return authenticated_user

def check_data(data):
    # Vérification que les données récupérée soit cohérente avec celle que nous avons observées
    error = []
    if data['meteo'] in ['clear','cloudy', 'rainy', 'snowy']:
        meteo = ['clear','cloudy', 'rainy', 'snowy'].index(data['meteo'], 1, 4) # en supposant qu'une météo soit entrée parmis les 4 possible on ressort avec son mode
    else: error.append("Wrong Meteo")
    if data['day'] in ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]:
        day = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"].index(data['day'], 0, 6) #en supposant que qu'un jour de la semaine est entrée, on ressort avec son mode
    else: error.append("Wrong WeekDay")
    if -10.7 < data['temp'] < 39.4: #ici j'ai pris les min et max présent dans le DF bike
        temp = data['temp']
    else: error.append("Wrong Temperature")
    if 22 < data['bike'] < 8714: #idem
        bike = data['bike']
    else: error.append("Wrong bike number") #Si la vérification se conclue par une erreur alors ce sera la valeur de daily
    daily = [meteo, temp, bike, day] #les données sont rangée dans le même ordre que les colonnes du DF bike
    return daily, error

def metrics(y,y_pred):
    # MSE 
    mse = mean_squared_error(y,y_pred,squared=True)
    # RMSE
    rmse = mean_squared_error(y,y_pred,squared=False)
    # MAE
    mae = mean_absolute_error(y,y_pred)
    # R2
    r2 = r2_score(y,y_pred)
    return  mse, rmse, mae, r2

@app.route("/status")
def status():
#Renvoie 1 si l'API fonctionne
    return "1\n"

@app.route("/permissions",methods=["POST"])
def permissions():
    data=request.get_json()
    if authenticate_user(data['username'],data['password'])==True:
        return "Good Id"
    else:
        raise Unauthorized("Wrong Id")

@app.route('/biketomorrow/LR',methods=["POST"])
def biketomorrow_LR():
    data=request.get_json() #data atendu exemple : {'username':'Quinlan','password':5210,'meteo':'rainy','temp':10,'bike':327,'day':'Wednesday'}
    if authenticate_user(data['username'],data['password'])==True:
        daily,error = check_data(data)
        ar = np.array([daily])
        X_test_user = pd.DataFrame(ar, columns=['mean_weathersit','mean_atemp', 'sum_cnt', 'weekday'])
        if len(error) == 0:
            # Entrainement du modèle linéaire
            lr = LinearRegression()
            # Entrainement du modèle
            Yfit_lr = lr.fit(X_train,y_train)
            # Prédiction de la variable cible pour le jeu de données TEST_user
            y_pred_lr = Yfit_lr.predict(X_test_user)
            return "There will be {} bike predicted tomorrow with the Linear Regression model".format(y_pred_lr)
        else: return error
    else:
        raise Unauthorized("Wrong Id")

@app.route('/metrics/LR', methods=["POST"])
def metrics_LR():
    data=request.get_json()
    if authenticate_user(data['username'],data['password'])==True:
        daily,error = check_data(data)
        ar = np.array([daily])
        X_test_user = pd.DataFrame(ar, columns=['mean_weathersit','mean_atemp', 'sum_cnt', 'weekday'])
        if len(error) == 0:
            # Entrainement du modèle linéaire
            lr = LinearRegression()
            # Entrainement du modèle
            Yfit_lr = lr.fit(X_train,y_train)
            # On prédit les y à partir de X_test
            y_pred_test=Yfit_lr.predict(X_test)
            # On prédit les y à partir de X_train
            y_pred_train=Yfit_lr.predict(X_train)
            # affichage des metrics
            mse_train, rmse_train, mae_train, r2_train = metrics(y_train, y_pred_train)
            mse_test, rmse_test, mae_test, r2_test = metrics(y_test, y_pred_test)
            output = '''
            The model Linear Regression metrics for train data: 
            MSE : {}
            RMSE : {}
            MAE : {}
            R2 : {}
            The model Linear Regression metrics for test data:
            MSE : {}
            RMSE : {}
            MAE : {}
            R2 : {}
            '''
            return output.format(mse_train, rmse_train, mae_train, r2_train, mse_test, rmse_test, mae_test, r2_test)
        else: return error
    else:
        raise Unauthorized("Wrong Id")
        
@app.route('/biketomorrow/LOGR',methods=["POST"])
def biketomorrow_LGST():
    data=request.get_json() #data atendu exemple : {'username':'Quinlan','password':5210,'meteo':'rainy','temp':10,'bike':327,'day':'Wednesday'}
    if authenticate_user(data['username'],data['password'])==True:
        daily,error = check_data(data)
        ar = np.array([daily])
        X_test_user = pd.DataFrame(ar, columns=['mean_weathersit','mean_atemp', 'sum_cnt', 'weekday'])
        if len(error) == 0:
            # Entrainement du modèle logistique regression
            logR = LogisticRegression(solver = "newton-cg")
            # Entrainement du modèle
            Yfit_logR = logR.fit(X_train, y_train)
            # Prédiction de la variable cible pour le jeu de données TEST_user
            y_pred_logR = Yfit_logR.predict(X_test_user)
            return "There will be {} bike predicted tomorrow with the Logistic Regression model".format(y_pred_logR)
        else: return error
    else:
        raise Unauthorized("Wrong Id")    

                
@app.route('/biketomorrow/KNR',methods=["POST"])
def biketomorrow_KGH():
    data=request.get_json() #data atendu exemple : {'username':'Quinlan','password':5210,'meteo':'rainy','temp':10,'bike':327,'day':'Wednesday'}
    if authenticate_user(data['username'],data['password'])==True:
        daily,error = check_data(data)
        ar = np.array([daily])
        X_test_user = pd.DataFrame(ar, columns=['mean_weathersit','mean_atemp', 'sum_cnt', 'weekday'])
        if len(error) == 0:
            # Entrainement du modèle logistique regression
            knR = KNeighborsRegressor(n_neighbors = 5)
            # Entrainement du modèle
            Yfit_knR = knR.fit(X_train, y_train)
            # Prédiction de la variable cible pour le jeu de données TEST_user
            y_pred_knR = Yfit_knR.predict(X_test_user)
            return "There will be {} bike predicted tomorrow with the KNeighbor regression model".format(y_pred_knR)
        else: return error
    else:
        raise Unauthorized("Wrong Id")

        

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
