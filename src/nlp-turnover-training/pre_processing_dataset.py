import pandas as pd
import csv
from google.cloud import translate

def translate_training_dataset_to_english():
    en_training_csv = open('/data/en_training_dataset.csv', 'w+', encoding="utf-8")
    csv_writer = csv.writer(en_training_csv, delimiter=',', lineterminator='\n')
    translator = translate.Client()
    portguese_dataset = pd.read_csv('/data/pt_training_dataset.csv', encoding='utf8')
    csv_writer.writerow(["review", "aspect", "aspect_category", "sentiment"])
    for reg in portguese_dataset.itertuples():
        pro = translator.translate(reg.pros, target_language='en')
        cons = translator.translate(reg.cons, target_language='en')
        csv_writer.writerow([pro['translatedText'], "NA", "NA", "positive"])
        csv_writer.writerow([cons['translatedText'], "NA", "NA", "negative"])

def translate_final_dataset_to_english():
    en_dataset_csv = open("en_experiment_dataset.csv", 'w+', encoding="utf8")
    data_writer = csv.writer(en_dataset_csv, delimiter=',', lineterminator="\n")
    count = 0
    translator = translate.Client()
    pt_dataset = pd.read_csv('final_dataset.csv', encoding='utf8')
    data_writer.writerow(["review", "former_emp"])
    for reg in pt_dataset.itertuples():
        pro = translator.translate(reg.pros, target_language='en')
        cons = translator.translate(reg.cons, target_language='en')
        data_writer.writerow([pro['translatedText'], reg.former_emp])
        data_writer.writerow([cons['translatedText'], reg.former_emp])
        count +=1
        print(count)

print("--- Final Dataset ----")
translate_final_dataset_to_english()