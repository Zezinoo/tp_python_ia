mes_temperatures = [23.5 , 24.9 , 22.8 , 23.1]

mes_temperatures.append(24.3)

max = max(mes_temperatures)
mean = sum(mes_temperatures)/len(mes_temperatures)
min = min(mes_temperatures)

print(f"Temp MAX : {max}\n Temp Moyenne : {mean:.1f} \n Temp MIN : {min}")
