temp_path = "../data/capteur_temp.txt"
press_path = "../data/capteur_press.txt"
hum_path = "../data/capteur_hum.txt"

paths = [temp_path, press_path , hum_path]

min_lines =[ sum(1 for _ in open(p)) for p in paths]
min_lines = min(min_lines)

chiffres = tuple([str(i) for i in range(0,10)])

measures = {}

try:
    with open(temp_path, "r") as f:
        iter = 1
        while iter <= min_lines:
            for line in f:
                if line.startswith(chiffres):
                    temp = float(line)
                    measures[iter] = {"temp" : temp}
                else:
                    msg = f"Line {iter} doesn't have a valid numeric value in temperature file! Entry ignored."
                    print(msg)
                iter += 1 
except FileNotFoundError:
    msg = "File not found on the specified path! Verify the inserted path."
    exit(msg)

try:
    with open(press_path, "r") as f:
        iter = 1
        while iter <= min_lines:
            for line in f:
                if line.startswith(chiffres):
                    v = float(line)
                    measures[iter]["press"] = v
                else:
                    msg = f"Line {iter} doesn't have a valid numeric value in pressure file! Entry ignored."
                    print(msg)
                iter += 1 
except FileNotFoundError:
    msg = "File not found on the specified path! Verify the inserted path."
    exit(msg)

try:
    with open(hum_path, "r") as f:
        iter = 1
        while iter <= min_lines:
            for line in f:
                if line.startswith(chiffres):
                    v = float(line)
                    measures[iter]["hum"] = v
                else:
                    msg = f"Line {iter} doesn't have a valid numeric value in humidity file! Entry ignored."
                    print(msg)
                iter += 1 
except FileNotFoundError:
    msg = "File not found on the specified path! Verify the inserted path."
    exit(msg)

print(str(measures))