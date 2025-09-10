capteur = {'id': 'C001' ,
           'type' : 'Température',
           'valeur' : 23.5}

capteur['valeur'] = 24.0

capteur['unite']= '°C'

print("INFOS DE CAPTEUR:")

print(f"Id de capteur : {capteur['id']} \n Type : {capteur['type']} \n Valeur : {capteur['valeur']} \n Unite : {capteur['unite']}")