import os
import json
from collections import Counter

import numpy as np

folder_name = "dane"
folder_path = os.path.join(os.getcwd(), folder_name)
files = os.listdir(folder_path)
files.remove("Churn_Modelling.csv")
files.remove("_info-data-discrete.txt")
systems = [files[j] for j in range(len(files)) if j % 2]
systems_types = [files[k] for k in range(len(files)) if not k % 2]

intents = {'intents': []}

for i in range(len(systems)):
    system = np.loadtxt('dane/' + systems[i], dtype=str)
    system_types = np.loadtxt('dane/' + systems_types[i], dtype=str)

    tags = []
    patterns = []
    responses = []

    if i == 0:
        tags.append("Powitanie")
        patterns.append(["Cześć!", "Dzień dobry", "Witam!", "Witaj!", "Co u ciebie słychać?", "Jak się masz?"])
        responses.append(["Cześć!", "Witaj w czym mogę pomóc?", "Dzień dobry! Co Cię tu przywiodło?"])

        tags.append("Pożegnanie")
        patterns.append(["Do widzenia", "Dobranoc", "Trzymaj się!"])
        responses.append(["Do widzenia", "Dobranoc", "Trzymaj się ciepło", "Trzymaj się"])

    obiekty = system.shape[0]

    tags.append("Obiekty_"+systems[i][:-4])
    patterns.append(["ile obiektów "+systems[i][:-4], "Jaka jest ilość obiektów w "+systems[i][:-4], "Jaka jest liczba obiektów "+systems[i][:-4]])
    responses.append(["W tym systemie decyzyjnym jest {} obiektów.".format(obiekty), "{} obiektów.".format(obiekty), "Ilość obiektów wynosi {}.".format(obiekty)])

    counts = Counter(system[:, system.shape[1]-1])
    classes = ""
    for key, value in counts.items():
        classes += " " + key + ": " + str(value)

    tags.append("Klasy_"+systems[i][:-4])
    patterns.append(["Ile klas "+systems[i][:-4], "Ile klas decyzyjnych "+systems[i][:-4], "Wielkość każdej klasy decyzyjnej "+systems[i][:-4], "Ile obiektów mają klasy decyzjyne "+systems[i][:-4], "Ile obiektów mają klasy decyzjyne w "+systems[i][:-4]])
    responses.append(["W tym systemie {} są {} klasy decyzyjne. Oto występujące klasy decyzjyne oraz ich wielkość\n{}.".format(systems[i][:-4], len(counts), classes), "Ilość klas dycyzyjnych {} wynosi {}. Wielkość poszczególnych klas to{}".format(systems[i][:-4], len(counts), classes), "Klasy decyzyjne w tym sytemie {} mają{} obiektów. Wszystkich klas decyzyjnych jest {}".format(systems[i][:-4], classes, len(counts))])

    diff_value_count = {}
    diff_value = {}
    common_value = {}
    min_value = {}
    max_value = {}
    attributes = system_types[:, 0]
    attributes_types = system_types[:, 1]
    for z in range(len(attributes)):
        diff_value_count[attributes[z]] = len(set(system[:, z]))
        diff_value[attributes[z]] = set(system[:, z])
        counts = Counter(system[:, z])
        common_value[attributes[z]] = counts.most_common()[0][0]
        if attributes_types[z] == "n":
            min_value[attributes[z]] = sorted(list(counts.values()))[0]
            max_value[attributes[z]] = sorted(list(counts.values()))[-1]
        else:
            min_value[attributes[z]] = "Atrybut symboliczny"
            max_value[attributes[z]] = "Atrybut symboliczny"

    diff = ""
    diff_count = ""
    common = ""
    min_max = ""
    for key, value in diff_value.items():
        diff += " " + key + ": "
        for v in value:
            diff += v + ", "
        diff += "\n"
    for key, value in diff_value_count.items():
        diff_count += " " + key + ": " + str(value) + "\n"
    for key, value in common_value.items():
        common += " " + key + ": " + str(value) + "\n"
    for key in min_value.keys():
        min_max += " " + key + ": (Min) " + str(min_value[key]) + " (Max) " + str(max_value[key]) + "\n"

    tags.append("Unikatowe_wartości_"+systems[i][:-4])
    patterns.append(["Jakie są unikatowe wartości w atrybutach "+systems[i][:-4], "Unikatowe wartości atrybutów "+systems[i][:-4]])
    responses.append(["Unikatowe wartości w poszczególnych atrybutach:\n{}".format(diff), "Wartości unikatowe atrybutów to:\n{}".format(diff)])

    tags.append("Ilość_unikatowych_wartości_"+systems[i][:-4])
    patterns.append(["Jaka jest ilość unikatowych wartości w atrybutach "+systems[i][:-4], "ilość unikatowych wartości atrybutów "+systems[i][:-4]])
    responses.append(["Ilość Unikatowych wartości w poszczególnych atrybutach:\n{}".format(diff_count), "Ilość wartości unikatowcyh atrybutów to:\n{}".format(diff_count)])

    tags.append("częste_wartości_"+systems[i][:-4])
    patterns.append(["Jakie wartości występują najcześciej w atrybutach "+systems[i][:-4], "najczęstrze wartości atrybutów "+systems[i][:-4]])
    responses.append(["Najczęstrze wartości w poszczególnych atrybutach:\n{}".format(common), "Wartości najczęscie występujące w atrybutach to\n:{}".format(common)])

    tags.append("min_max_"+systems[i][:-4])
    patterns.append(["Jakie są minimalne i maksymalne wartości "+systems[i][:-4], "Jakie sa minimalne oraz maksymalne wartośći dla atrybutów "+systems[i][:-4]])
    responses.append(["Minimalne i maksymalne wartości w poszczególnych atrybutach:\n{}".format(min_max)])

    for i in range(len(tags)):
        intents['intents'].append({"tag": tags[i], "patterns": patterns[i], "responses": responses[i]})


with open('intents.json', 'w') as fp:
    json.dump(intents, fp, indent=4, ensure_ascii=False)
