

file_path = "../data/capteur_temp.txt"
temp_list = []
chiffres = tuple([str(i) for i in range(0,10)])

try:
    with open(file_path, "r") as f:
        iter = 1
        for line in f:
            if line.startswith(chiffres):
                temp_list.append(float(line))
            else:
                msg = f"Line {iter} doesn't have a valid numeric value! Entry ignored."
            iter += 1 
except FileNotFoundError:
    msg = "File not found on the specified path! Verify the inserted path."
    exit(msg)


print(f"MAX : {max(temp_list):.2f}")
print(f"MIN : {min(temp_list):.2f}")
print(f"MEAN : {sum(temp_list)/len(temp_list):.2f}")
