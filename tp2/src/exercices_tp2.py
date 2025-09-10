#!/usr/bin/env python3
# Importez toutes les bibliothèques nécessaires (numpy, matplotlib,
# etc.)
import numpy as np
from matplotlib import pyplot as plt

# Définissez éventuellement des fonctions



class Exercise1():
    def __init__(self):
        self.show_graph()

    def f(self, t : np.ndarray , A = 5 , alpha = 0.5 , omega = 2*np.pi) -> np.ndarray:
        result = A * np.exp(-alpha * t) * np.cos(omega*t)
        return result

    def show_graph(self):
        t = np.linspace(0,50 , 501)
        y = self.f(t)
        rng = np.random.default_rng(13)
        bruit = rng.normal(0 , 0.2 , size = t.shape)
        y_bruitee =  y + bruit
        y_mean = np.mean(y_bruitee)
        y_stdev = np.std(y_bruitee)
        y_min = np.min(y_bruitee)
        y_max = np.max(y_bruitee)

        mask_low_value = np.abs(y_bruitee) < 0.1
        low_value_points = y_bruitee[mask_low_value]
        percentage_of_low_value = len(low_value_points)/len(y_bruitee)

        signs = np.sign(y)
        diffs = np.diff(signs)
        
        idxs = np.argwhere(np.abs(diffs) == 2)    
        
        plt.plot(t , y_bruitee , label = f"Mean : {y_mean :.2f} , StDev : {y_stdev:.2f} , Min : {y_min:.2f} , Max : {y_max :.2f} , Percentage : {percentage_of_low_value:.2f}")
        plt.title('Cos Function Bruitee')
        plt.grid(True)

        idxs_next = idxs + 1
        #idxs = np.concat([idxs, idxs_next])
        idxs = np.unique(idxs)
        xpoints = t[idxs]


        interpolations = np.interp(xpoints , t , y)
        print(interpolations)

        plt.plot(xpoints, interpolations, '-x')
        
        plt.legend()
        plt.show()

class Exercice3():
    def __init__(self):
        self.rng = np.random.default_rng(13)
        self.show_graph()

    def show_graph(self):

        db1 = self.rng.normal(0,0.2,1000)
        db2 = self.rng.normal(0,0.2,1000)

        plt.hist(db1 , bins = 40 , color="blue" , alpha = 0.5 , label= "DB1")
        plt.hist(db2 , bins = 40 , color = "red" , alpha = 0.5 , label = "DB2")
        plt.title("Two Gaussian Distributions")
        plt.legend()
        plt.show()

class Exercice4():
    def __init__(self):
        self.show_graph()
    def show_graph(self):
        x = np.random.uniform(0,20,100)
        y = np.random.uniform(0,20,100)

        plt.scatter(x,y,cmap='viridis' , c = y)
        plt.colorbar()
        plt.show()

class Exercice5():
    def __init__(self):
        self.show_graph()

    def show_graph(self):
        fig , axs = plt.subplots(2,2 , sharex=True)
        n_cols = axs.shape[1]
        n_rows = axs.shape[0]
        size = 100
        data_set = {i : (np.random.random(size) , np.random.random(size)) for i in range(len(axs.flatten()))}
        row = 0
        for i in range(len(axs.flatten())):
            col = i%n_cols
            if i%n_rows == 0 and i != 0:
                row = row + 1
            axs[row][col].scatter(data_set[i][0] , data_set[i][1])
            axs[row][col].axis("off")
        
        plt.show()

class Exercice6():
    def __init__(self):
        self.show_graph()
    
    def show_graph(self):
        x = np.linspace(0 , 5*np.pi , 500)
        sinx = np.sin(x)
        sin2x = np.sin(2*x)
        sin3x = np.sin(3*x)


        plt.plot(x , sinx , color = "b" , label = r"$\sin(x)$")
        plt.plot(x , sin2x , color = "r" , label = r"$\sin(2x)$")
        plt.plot(x , sin3x , color = "g" , label = r"$\sin(3x)$")

        plt.legend()
        plt.show()

class Exercice7():
    def __init__(self):
        self.show_graph()

    def temp(self , t):
        return 300 + 10*np.sin(0.1*t)
    
    def press(self , t):
        return 100 + 5*np.cos(0.1*t)

    def show_graph(self):
        t = np.linspace(0,100 , 500)

        temp = self.temp(t)
        press = self.press(t)

        fig , axs = plt.subplots(2 , 1 , sharex= True)

        axs[0].plot(t , temp , label = "Temp")
        axs[0].set_ylabel("Temp")

        axs[1].plot(t , press , label = "Press")
        axs[1].set_ylabel("Press")
        axs[1].set_xlabel("Time (s)")

        plt.show()

class Exercice8():
    def __init__(self):
        self.show_graph()
    
    def show_graph(self):
        x = np.linspace(-10 , 10 , 500)
        sigmoid = 1/(1 + np.exp(-x))

        x_inflection = 0
        y_inflection = 0.5

        plt.plot(x , sigmoid)
        plt.annotate(f'Inflection Point\n({x_inflection:.2f}, {y_inflection:.2f})', xy=(x_inflection, y_inflection),
                xytext=(x_inflection+0.5, y_inflection+0.5), arrowprops=dict(facecolor='black', arrowstyle='->'),
                fontsize=10)
        
        plt.savefig('data/out/exercic8.png', dpi=300)
        plt.legend()
        plt.show()
 
class Exercice9():
    def __init__(self):
        self.show_graph()

    def temp(self , t , b):
        return 20 + 2 * np.sin(2 * np.pi * t / 120) + b
    
    def press(self , t , temp , b):
        return 101 + 0.8 * np.cos(2* np.pi * t / 90) + 0.02 * temp + b
    
    def humidite(self , t , temp , phi , b):
        return 50 + 5 * np.sin(2* np.pi*t / 150 + phi) - 0.1 * temp + b
    
    def show_graph(self):
        phi = 1/4 * np.pi
        t = np.linspace(0 , 300 , 601)
        bruits = [np.random.uniform(len(t)) for i in range(3)]

        temp = self.temp(t , bruits[0])
        press = self.press(t , temp , bruits[1])
        humidite = self.humidite(t , temp , phi , bruits[2] )

        graphs = [temp , press , humidite]
        ylabel = ["Temperature" , "Pression" , "Humidite"]        

        fig , axs = plt.subplots(3 , 1 , sharex= True)

        for i in range(len(axs)):
            axs[i].plot(t , graphs[i])
            axs[i].set_ylabel(ylabel[i])
        
        plt.legend()
        plt.show()

class Exercice10():
    def __init__(self):
        self.show_graph()

    def show_graph(self):
        rng = np.random.default_rng(17) # générateur déterministe
        t = np.linspace(0, 100, 500)
        temp = 20 + 2 * np.sin(2*np.pi*t/50) + rng.normal(0, 0.5, size=t.shape)
        pres = 101 + 0.5 * np.cos(2*np.pi*t/40) + 0.05 * temp + rng.normal(0, 0.2,size=t.shape)
        humi = 50 + 5 * np.sin(2*np.pi*t/70 + np.pi/6) - 0.1 * temp + rng.normal(0,0.5, size=t.shape)
        vent = 10 + 0.3 * pres + rng.normal(0, 1.0, size=t.shape)


        arrs = [temp , pres , humi , vent]

        stacked = np.vstack(tuple(arrs))
        corr_matrix = np.corrcoef(stacked)

        labels = ["Temp", "Pres", "Humi", "Vent"]
        plt.xticks(range(len(labels)), labels)
        plt.yticks(range(len(labels)), labels)

        im = plt.imshow(corr_matrix , cmap = "coolwarm")
        plt.colorbar(im)
        plt.show()

class Exercice11():
    def __init__(self):
        self.show_graph()

    def show_graph(self):
        from matplotlib.patches import Rectangle, Circle
    
        fig, axs = plt.subplots(2, 1, sharex=True, figsize=(8, 6))

        
        x = np.linspace(0, 2*np.pi, 1000)
        y_sin = np.sin(x)
        y_cos = np.cos(x)

        
        axs[0].plot(x, y_sin, label="sin(x)")
        axs[0].set_title("Sinus")
        axs[0].set_ylabel("Amplitude")
        axs[0].legend()

       
        x0 = np.pi/2
        y0 = np.sin(x0)  # = 1
        rect_width = 0.6        
        rect_height = 0.4        
        rect = Rectangle((x0 - rect_width/2, y0 - rect_height/2),
                        rect_width, rect_height,
                        fill=False, linewidth=2)
        axs[0].add_patch(rect)

        axs[1].plot(x, y_cos, label="cos(x)")
        axs[1].set_title("Cosinus")
        axs[1].set_xlabel("x (radians)")
        axs[1].set_ylabel("Amplitude")
        axs[1].legend()

        xc = np.pi
        yc = np.cos(xc)  # = -1
        circle = Circle((xc, yc), radius=0.3, fill=False, linewidth=2)
        axs[1].add_patch(circle)

        plt.tight_layout()
        plt.show()




def main(A = 5  , alpha = 0.5 , omega = 2*np.pi):
    e1 = Exercise1()
    e3 = Exercice3()
    e4 = Exercice4()
    e5 = Exercice5()
    e6 = Exercice6()
    e7 = Exercice7()
    e8 = Exercice8()
    e9 = Exercice9()
    e10 = Exercice10()
    e11 = Exercice11()
# Vos instructions
if __name__ == "__main__": 
    main()
    pass