import sys
import math 

filename = sys.argv[1]

totalInstances = 0
senseLabels = set()
foldSize = 25
with open(filename, "r") as file:
    for line in file:
        line = line.strip()
        if "senseid" in line:
            totalInstances += 1
            start = line.find("senseid") + len("senseid")
            end = line.find('"', start)
            sense = line[start:end]
            senseLabels.add(sense)
print(f"Total instances: {totalInstances}")
print(f"Possible sense labels: {len(senseLabels)}")

foldSize = math.ceil(totalInstances / 5)

lastFoldSize = totalInstances - (foldSize * 4)
print(f"fold 1: {foldSize}")
print(f"fold 2: {foldSize}")
print(f"fold 3: {foldSize}")
print(f"fold 4: {foldSize}")
print(f"fold 5: {lastFoldSize}")