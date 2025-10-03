#!/usr/bin/env python3
# Imports nécessaires (numpy, matplotlib, etc.)
# Fonctions utiles (si besoin)
def exercice_1():
    longeurs = [i for i in range(5,25,5)]
    carres = list(map(lambda x : x**2 , longeurs))
    sup_dix = list(filter(lambda x : x>10 , longeurs))
    from functools import reduce
    produit = reduce(lambda x , y : x*y , longeurs)
    return carres, sup_dix , produit

def exercice_2():
    ages =  [18, 21, 25, 30, 35]
    ages_en_mois = [a*12 for a in ages]
    print(ages_en_mois)

def gen_nombre_pairs(n):
    for i in range(n):
        if i %2 ==0 :
            yield i




def main():
    _ , _ ,c = exercice_1()
    print(c)
    pass
if __name__ == "__main__":
    main()
