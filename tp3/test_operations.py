#!/usr/bin/env python3
# Imports nécessaires (numpy, matplotlib, etc.)

from mon_paquetage.operations import addition , multiplication

# Fonctions utiles (si besoin)

def main(a,b):
    print(addition(a,b))
    print(multiplication(a,b))

if __name__ == "__main__":
    main(2,3)
