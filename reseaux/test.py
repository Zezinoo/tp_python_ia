from abc import ABC ,abstractmethod # ABC veut dire Abstract Base Classe
#Obligatoire pour pouvoir profiter du décorateur @abstractmethod

class Titi(ABC) :
    N=0
    def __init__(self) :
        Titi.N=Titi.N+1

    def getN () :
        return Titi.N

    def initialize (self) :
        Titi.N=0

    def __repr__(self) :
        return "Titi de norme  "+str(self.norme())

    @abstractmethod
    def norme(self) :
        pass


class Toto (Titi) :
    def __init__(self, x,y) :
        super().__init__()
        self.__X, self.__Y = x , y

    def initialize (self, x,y) :
        self.__X, self.__Y = x , y

    def getXY(self) :
        return(self.__X,self.__Y)

    def norme(self) :
        return self.__X**2+self.__Y**2






print(Titi.N) 