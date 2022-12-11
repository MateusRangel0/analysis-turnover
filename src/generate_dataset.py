import csv

visual_turnover = open('turnover_sentiment.csv', 'w+', encoding='utf8')
visual_csv_writer = csv.writer(visual_turnover, delimiter=',', lineterminator='\n')

def create_columns():
    visual_csv_writer.writerow(["sentiment", "former_emp"])

def generate_visual_dataset(sentiment, former_emp):
    visual_csv_writer.writerow([sentiment, former_emp])