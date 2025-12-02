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
        std_dev = np.std(temperatures , ddof = 1) # Populational stdev
        max = np.max(temperatures)
        min = np.min(temperatures)
        df_capteurs = pd.DataFrame(data=
                                   {"id_capteur" : ['C001', 'C002', 'C003', 'C004', 'C005'],
                                    "type" : ['Température', 'Pression', 'Humidité', 'Température', 'Pression'],
                                    "valeur" : [22.5, 1013.0, 45.0, 23.1, 1011.0],
                                    "unité" : ['°C', 'hPa', '%', '°C', 'hPa']})
        print("## Affichage de premieres lignes df_capteurs ##")
        print(df_capteurs.head())
        print("## Affichage capteurs temperature ##")
        print(df_capteurs[df_capteurs["type"] == "Température"])
        mean_by_type = df_capteurs.groupby(["type" , "unité"])["valeur"].mean()
        print("## Affichage de moyenne de valeurs par type ##")
        print(mean_by_type)

    def ex_2(self):
        df = pd.read_csv("data/sensors1.csv")
        print('## Affichage des donnes generales ##')
        print(df.info())
        print(df.describe())
        df['heure'] = pd.to_datetime(df['heure'], format='%H:%M:%S')
        df = df.fillna(df.mean())
        print('## Verification valeures manquantes ##')
        print(df.isna().sum())
        print(df.tail())

        fig , ax = plt.subplots( 2 , 2 , figsize = (10,4))


        ax[0][0].plot(df["heure"] , df["temperature"]) 
        ax[0][0].set_xlabel("Heure")
        ax[0][0].set_ylabel('Temperature C°')
        ax[0][1].hist(df["pression"])
        ax[0][1].set_xlabel("Pression hPa")
        ax[0][1].set_ylabel('N')
        ax[1][0].scatter(df["temperature"] , df["humidite"])
        ax[1][0].set_xlabel("Temperature C°")
        ax[1][0].set_ylabel('Humidite %')

        corr = df[["temperature", "pression", "humidite"]].corr()

        cax = ax[1][1].imshow(corr, cmap='coolwarm', vmin=-1, vmax=1)

        # labels
        ax[1][1].set_xticks(np.arange(len(corr.columns)))
        ax[1][1].set_yticks(np.arange(len(corr.columns)))
        ax[1][1].set_xticklabels(corr.columns)
        ax[1][1].set_yticklabels(corr.columns)

        plt.setp(ax[1][1].get_xticklabels(), rotation=45, ha="right")

        for i in range(len(corr)):
            for j in range(len(corr)):
                ax[1][1].text(j, i, f"{corr.iloc[i, j]:.2f}",
                            ha="center", va="center")

        # add colorbar
        plt.colorbar(cax, ax=ax[1][1])

        ax[1][1].set_title("Matrice Correlation")
        


        print(corr)
        plt.tight_layout()
        fig.text(
    0.5, 0.02,
    "2.7 : Les variables les plus corrélées sont la température et l’humidité.\n"
    "Elles sont corrélées négativement.\n"
    "Physiquement, cela s’explique par le fait que lorsqu’on a une température plus élevée,\n"
    "il y a davantage d’évaporation dans ce climat, et donc moins d’humidité dans l’air.",
    ha="center", fontsize=12
)

        plt.show()
    
    def ex_3(self):
        features = self.sensor_1[["temperature" , "pression"]]
        target = self.sensor_1[["humidite"]]
        X_train , X_test , y_train , y_test = train_test_split(features , target , test_size=0.2 , random_state= 42)
        model = LinearRegression()
        model.fit(X_train , y_train)

        print("Coefficients :", model.coef_)
        print("Ordonnée à l’origine (intercept) :", model.intercept_)

        predictions = model.predict(X_test)


        rmse = np.sqrt(mean_squared_error(y_test, predictions)) # take square root because this func returns the error squared
        r2 = r2_score(y_test , predictions)

        print(f"RMSE : {rmse}")
        print(f"R^2 : {r2}")

        x_line = np.linspace(y_test.min(), y_test.max(), 200)

        plt.scatter(y_test, model.predict(X_test),
                    label=f'RMSE : {rmse:.2f}\nR^2 : {r2:.2f}')

        plt.plot(x_line, x_line, color="black", label='y=x')

        plt.xlabel("Valeurs Cibles" , fontsize = 14)
        plt.ylabel("Valeur Predictions" , fontsize = 14)

        reponse = (
        "3.8. Le modèle parvient à capturer une tendance générale entre la température et l’humidité, "
        "mais la qualité des prédictions reste faible, comme le montre le R² bas. On observe une dispersion notable autour "
        "de la diagonale, ce qui révèle des erreurs systématiques, avec des valeurs parfois sous-estimées ou surestimées. \n "

        "3.9. Les performances sont influencées par plusieurs facteurs : la relation entre température et humidité n’est pas "
        "strictement linéaire, les mesures contiennent du bruit, certaines variables importantes comme le vent ou "
        "l’ensoleillement ne sont pas prises en compte, et la pression apporte peu d’information utile, ce qui a été montré "
        "par sa faible corrélation. La taille limitée du jeu de données réduit aussi la précision du modèle."
        )

        plt.tight_layout()

        plt.subplots_adjust(bottom=0.30)
        plt.gcf().text(0.5, 0.05, reponse, ha='center', va='center', fontsize=12, wrap=True)

        plt.legend()
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

        rmse = np.sqrt(mean_squared_error(y_test , y_pred_se ))
        r2 = r2_score(y_test , y_pred_se)

        print(f"RMSE : {rmse}")
        print(f"R^2 : {r2}")

        print("Coefficients :", model.coef_)
        print("Ordonnée à l’origine (intercept) :", model.intercept_)
    

        fig , axs = plt.subplots(2,2)

        # Pression non standardisée
        axs[0][0].hist(X_train["pression"], bins=10,
                    label=rf"$\sigma = {np.std(X_train['pression'], ddof=1):.2f}$")
        axs[0][0].set_title("Pression non standardisée")
        axs[0][0].legend()

        # Pression standardisée
        axs[0][1].hist(X_train_s[:, 1], bins=10,
                    label=rf"$\sigma = {np.std(X_train_s[:, 1]):.2f}$")
        axs[0][1].set_title("Pression standardisée")
        axs[0][1].legend()

        # Température non standardisée
        axs[1][0].hist(X_train["temperature"], bins=10,
                    label=rf"$\sigma = {np.std(X_train['temperature'], ddof=1):.2f}$")
        axs[1][0].set_title("Température non standardisée")
        axs[1][0].legend()

        # Température standardisée  
        axs[1][1].hist(X_train_s[:, 0], bins=10,
                    label=rf"$\sigma = {np.std(X_train_s[:, 0]):.2f}$")
        axs[1][1].set_title("Température standardisée")
        axs[1][1].legend()

        plt.tight_layout()

        reponse = (
        "4.4. Le resultats obtenues par RMSE et R^2 sont identiques car, meme si les donnes sont standardises, il s'agit encore"
        "de le meme jeu de donnes. \n "

        "4.5. La importance de standardiser les donnes c'est principalement de ameliores lea analyses statistiques " \
        "de la performance de chaque variable, car maintenant l'effet  d'echelle sur les jeux de donnés n'est pas un" \
        "facteur importante."
        )

        plt.subplots_adjust(bottom=0.30)
        plt.gcf().text(0.5, 0.05, reponse, ha='center', va='center', fontsize=12, wrap=True)


        plt.subplots_adjust(bottom=0.1)


        plt.show()

        x_line = np.linspace(target.min(), target.max(), 200).reshape(-1,1)
        plt.scatter(y_test , model.predict(X_test_s) , label = f"RMSE = {rmse:.2f}\n$R^2$ = {r2:.2f}")
        plt.plot(x_line , x_line , color = "black" , label = 'y=x')
        plt.xlabel("Valeurs Cibles")
        plt.ylabel("Valeur Predictions")
        plt.legend()
        plt.show()


    def ex_5(self):
        features = self.sensor_2[["temperature" , "pression"]]
        target = self.sensor_2[["humidite"]]
        X_train , X_test , y_train , y_test = train_test_split(features , target , test_size=0.2 , random_state= 42)
        model = LinearRegression()
        model.fit(X_train , y_train)

        predictions = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test , predictions ))
        r2 = r2_score(y_test , predictions)

        poly2 = Pipeline([
            ("poly", PolynomialFeatures(degree=2, include_bias=False)),
            ("scaler", StandardScaler()),
            ("linreg", LinearRegression())
            ])
        poly2.fit(X_train, y_train)
        y_pred_poly_train = poly2.predict(X_train)
        y_pred_poly_test = poly2.predict(X_test)
        rmse_poly_train = np.sqrt(mean_squared_error(y_train, y_pred_poly_train))
        rmse_poly_test = np.sqrt(mean_squared_error(y_test, y_pred_poly_test))
        r2_poly_train = r2_score(y_train, y_pred_poly_train)
        r2_poly_test = r2_score(y_test, y_pred_poly_test)


        print(f"RMSE poly2 train : {rmse_poly_train:.3f} | R^2 poly2 train : {r2_poly_train:.3f}")
        print(f"RMSE poly2 test : {rmse_poly_test:.3f} | R^2 poly2 test : {r2_poly_test:.3f}")

        x_line = np.linspace(target.min(), target.max(), 200).reshape(-1,1)
        plt.scatter(y_train , y_pred_poly_train , label = f"RMSE = {rmse_poly_train:.2f}\n$R^2$ = {r2_poly_train:.2f}")
        plt.plot(x_line , x_line , color = "black" , label = 'y=x')
        plt.xlabel("Valeurs Cibles")
        plt.ylabel("Valeur Predictions")
        plt.legend()

        plt.tight_layout()

        reponse = (
        "5.3 Par rapport à la prédiction obtenue avec une régression linéaire, la prédiction est "
        "nettement améliorée, ce qui se traduit par une amélioration d’environ 30 % sur le coefficient R². "
        "Cela confirme également l’hypothèse de relations non linéaires entre les variables explicatives et la cible. "
        "Malgré cela, la performance sur le jeu de données d’entraînement reste légèrement supérieure à celle du jeu "
        "de test, ce qui est attendu. Cependant, l’écart entre les performances des deux jeux reste faible, ce qui "
        "montre qu’il s’agit d’un modèle équilibré. Augmenter l’ordre du polynôme pourrait entraîner un surapprentissage, "
        "où le modèle s’ajuste trop au jeu de données d’entraînement et prédit moins bien les données de test. On voit justement ca avec" \
        "le test suivante avec une polynome d'ordre 10."
        )
        plt.subplots_adjust(bottom=0.30)
        plt.gcf().text(0.5, 0.05, reponse, ha='center', va='center', fontsize=10, wrap=True)


        plt.subplots_adjust(bottom=0.1)

        plt.show()


        poly10 = Pipeline([
            ("poly", PolynomialFeatures(degree=10, include_bias=False)),
            ("scaler", StandardScaler()),
            ("linreg", LinearRegression())
            ])
        poly10.fit(X_train, y_train)
        y_pred_poly_train = poly10.predict(X_train)
        y_pred_poly_test = poly10.predict(X_test)
        rmse_poly_train = np.sqrt(mean_squared_error(y_train, y_pred_poly_train))
        rmse_poly_test = np.sqrt(mean_squared_error(y_test, y_pred_poly_test))
        r2_poly_train = r2_score(y_train, y_pred_poly_train)
        r2_poly_test = r2_score(y_test, y_pred_poly_test)


        print(f"RMSE poly4 train : {rmse_poly_train:.3f} | R^2 poly4 train : {r2_poly_train:.3f}")
        print(f"RMSE poly4 test : {rmse_poly_test:.3f} | R^2 poly4 test : {r2_poly_test:.3f}")


if __name__ == "__main__":
    exs_executed = [5]

    exs = Exercices()

    for n in exs_executed:
        method_name = f"ex_{n}"
        if hasattr(exs , method_name):
            getattr(exs , method_name)()
