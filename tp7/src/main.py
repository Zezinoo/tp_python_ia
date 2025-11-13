import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error , r2_score
from sklearn.preprocessing import StandardScaler , PolynomialFeatures
from sklearn.pipeline import Pipeline

class Exercices():
    def __init__(self):
        self.sensor_1 = self.read_sensor_1()   
        self.sensor_2 = self.read_sensor_2()     

    def read_sensor_2(self):
        df = pd.read_csv("data/sensors2.csv")
        df['heure'] = pd.to_datetime(df['heure'], format='%H:%M:%S')
        df = df.fillna(df.mean())
        return df


    def read_sensor_1(self):
        df = pd.read_csv("data/sensors1.csv")
        df['heure'] = pd.to_datetime(df['heure'], format='%H:%M:%S')
        df = df.fillna(df.mean())
        return  df


    def ex_1(self):
        temperatures = np.array([22.5, 23.0, 22.8, 23.1,22.9, 22.8, 22.7])
        mean = np.mean(temperatures)
        std_dev = np.std(temperatures)
        max = np.max(temperatures)
        min = np.min(temperatures)
        df_capteurs = pd.DataFrame(data=
                                   {"id_capteur" : ['C001', 'C002', 'C003', 'C004', 'C005'],
                                    "type" : ['Température', 'Pression', 'Humidité', 'Température', 'Pression'],
                                    "valeur" : [22.5, 1.02, 45.0, 23.1, 1.01],
                                    "unité" : ['°C', 'bar', '%', '°C', 'bar']})
        
        print(df_capteurs.head())
        print("##")
        print(df_capteurs[df_capteurs["type"] == "Température"])
        mean_by_type = df_capteurs.groupby(["type" , "unité"])["valeur"].mean()
        print("##")
        print(mean_by_type)

    def ex_2(self):
        df = pd.read_csv("data/sensors1.csv")
        print(df.info())
        print(df.describe())
        df['heure'] = pd.to_datetime(df['heure'], format='%H:%M:%S')
        df = df.fillna(df.mean())
        print(df.tail())

        fig , ax = plt.subplots( 2 , 2 , figsize = (10,4))


        ax[0][0].plot(df["heure"] , df["temperature"]) # This graph is strange because the temperatures loops back
        ax[0][1].hist(df["pression"])
        ax[1][0].scatter(df["temperature"] , df["humidite"])

        corr = df[["temperature" , "pression" , "humidite"]].corr()
        print(corr)
        plt.show()
    
    def ex_3(self):
        features = self.sensor_1[["temperature" , "pression"]]
        target = self.sensor_1[["humidite"]]
        X_train , X_test , y_train , y_test = train_test_split(features , target , test_size=0.2 , random_state= 42)
        model = LinearRegression()
        model.fit(X_train , y_train)
        print(model.coef_)

        predictions = model.predict(X_test)


        rmse = mean_squared_error(y_test , predictions )
        r2 = r2_score(y_test , predictions)

        print(f"RMSE : {rmse}")
        print(f"R^2 : {r2}")

        #plt.scatter(features["temperature"], target, label="Temperature")
        #plt.scatter(features["pression"], target, label="Pression")
        x_line = np.linspace(target.min(), target.max(), 200).reshape(-1,1)
        plt.scatter(y_test , model.predict(X_test))
        plt.plot(x_line , x_line , color = "black")
        plt.xlabel("target")
        plt.ylabel("predictions")
        plt.show()

        pass

    def ex_4(self):
        features = self.sensor_1[["temperature" , "pression"]]
        target = self.sensor_1[["humidite"]]
        X_train , X_test , y_train , y_test = train_test_split(features , target , test_size=0.2 , random_state= 42)
        scaler = StandardScaler().fit(X_train)
        X_train_s = scaler.transform(X_train)
        X_test_s = scaler.transform(X_test)

        model = LinearRegression()
        model.fit(X_train_s , y_train)

        y_pred_se = model.predict(X_test_s)

        rmse = mean_squared_error(y_test , y_pred_se )
        r2 = r2_score(y_test , y_pred_se)

        print(f"RMSE : {rmse}")
        print(f"R^2 : {r2}")

        #Visualizing variance

        fig , axs = plt.subplots(2,2)
        axs[0][0].hist(X_train["pression"] , bins = 10)
        axs[0][1].hist(X_train_s[:,0] , bins = 10)

        axs[1][0].hist(X_train["temperature"] , bins = 10)
        axs[1][1].hist(X_train_s[:,1] , bins = 10)

        plt.show()

        x_line = np.linspace(target.min(), target.max(), 200).reshape(-1,1)
        plt.scatter(y_test , model.predict(X_test_s))
        plt.plot(x_line , x_line , color = "black")
        plt.xlabel("target")
        plt.ylabel("predictions")
        plt.show()


    def ex_5(self):
        features = self.sensor_2[["temperature" , "pression"]]
        target = self.sensor_2[["humidite"]]
        X_train , X_test , y_train , y_test = train_test_split(features , target , test_size=0.2 , random_state= 42)
        model = LinearRegression()
        model.fit(X_train , y_train)

        predictions = model.predict(X_test)


        rmse = mean_squared_error(y_test , predictions )
        r2 = r2_score(y_test , predictions)

        poly2 = Pipeline([
            ("poly", PolynomialFeatures(degree=2, include_bias=False)),
            ("scaler", StandardScaler()),
            ("linreg", LinearRegression())
            ])
        poly2.fit(X_train, y_train)
        y_pred_poly = poly2.predict(X_test)
        rmse_poly = mean_squared_error(y_test, y_pred_poly)
        r2_poly = r2_score(y_test, y_pred_poly)
        print(f"RMSE poly2 : {rmse_poly:.3f} | R^2 poly2 : {r2_poly:.3f}")




if __name__ == "__main__":
    exs_executed = [1,2,3,4,5]

    exs = Exercices()

    for n in exs_executed:
        method_name = f"ex_{n}"
        if hasattr(exs , method_name):
            getattr(exs , method_name)()
