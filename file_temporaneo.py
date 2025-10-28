import re

# Percorso del file di input
file_path = '/davinci-1/home/dmor/cifar100_all_plotaug.o1989808'

with open(file_path, 'r') as f:
    lines = f.readlines()

in_block = False
collect = False
TAG_values = []
TAW_values = []

for line in lines:
    line = line.strip()

    if re.match(r'^cifar100_', line):
        if collect:
            # Stampa i risultati senza doppie virgolette
            print("TAG = [" + ", ".join(f'"{v}"' for v in TAG_values) + "]")
            print("TAW = [" + ", ".join(f'"{v}"' for v in TAW_values) + "]")
            print()
            TAG_values = []
            TAW_values = []
            collect = False
        in_block = True

    if in_block and line.startswith("SEED:"):
        seed = int(line.split(":")[1].strip())
        collect = (seed == 9)

    if collect and "number of augmentation" in line:
        match = re.search(r"TAG = ([\d.]+ ± [\d.]+), TAW = ([\d.]+ ± [\d.]+)", line)
        if match:
            TAG_values.append(match.group(1))
            TAW_values.append(match.group(2))

# Stampa l'ultimo blocco
if collect and TAG_values:
    print("TAG = [" + ", ".join(f'"{v}"' for v in TAG_values) + "]")
    print("TAW = [" + ", ".join(f'"{v}"' for v in TAW_values) + "]")
