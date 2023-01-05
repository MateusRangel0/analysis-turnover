import os
import json
import csv
import ast

reviews_data = open('reviews.csv', 'w+', encoding="utf-8")

csv_writer = csv.writer(reviews_data, delimiter=',', lineterminator='\n')

empresas = ['accenture', 'stefanini', 'ibm', 'totvs', 'algar-tech', 'sonda-it', 'hp-inc', 'dell', 'linx', 'oracle', 'resource-it-solutions', 'ericsson', 'vivo-telefonica-brasil', 'tivit', 'tim', 'claro-brasil', 'b2w-digital', 'concentrix', 'nextel-telecomunicacoes']

count = 1
concentrix = 0
csv_writer.writerow(["user", "avaliation", "former_employeer"])
for root, dirs, files in os.walk('C:\\Users\\asana\\Workspace\\lovemondays-experiment\\generate-dataset-spreadsheet//'):
    for name in files:
        if name.endswith((".json")):
            for i in empresas:
                filename = "data\\" + i + "\\" + name
                actual_json_file = open(filename, 'r', encoding='utf-8')
                currentFile = json.load(actual_json_file)
                print("Processing file #" + str(count))
                count += 1
                if "Marketing da Concentrix" not in currentFile[1]["avaliation"]:
                    if  "Ex-funcionário" in currentFile[0]["user"]:
                        csv_writer.writerow([currentFile[0]["user"], currentFile[1]["avaliation"], True])
                    else:
                        csv_writer.writerow([currentFile[0]["user"], currentFile[1]["avaliation"], False])
                else:
                    concentrix += 1
reviews_data.close()
print("Done")
print("Deleted Rows")
print("Spamming: " + str(concentrix))