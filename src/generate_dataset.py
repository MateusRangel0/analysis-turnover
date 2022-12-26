import csv

turnover = open('turnover_sentiment.csv', 'w+', encoding='utf8')

visual_turnover = open('2visual_turnover.csv', 'w+', encoding='utf8')

csv_writer = csv.writer(turnover, delimiter=',', lineterminator='\n')

visual_csv_writer = csv.writer(visual_turnover, delimiter=',', lineterminator='\n')

def create_columns():
    visual_csv_writer.writerow(["sentiment", "aspect", "former_emp"])
    csv_writer.writerow(["pay", "conditions", "nature_of_work", "promotion", "supervision", "rewards", "coworkers", "communication", "benefits", "former_emp"])


def generate_visual_dataset(sentiment, aspect, former_emp):
    visual_csv_writer.writerow([sentiment, aspect, former_emp])

def generate_quantitative_dataset(sentiment, aspect, former_emp):
    if (aspect == 'pay'):
        csv_writer.writerow([sentiment, 0, 0, 0, 0, 0, 0, 0, 0,  former_emp])
    if (aspect == 'conditions'):
        csv_writer.writerow([0, sentiment, 0, 0, 0, 0, 0, 0, 0,  former_emp])
    if (aspect == 'nature_of_work'):
        csv_writer.writerow([0, 0, sentiment, 0, 0, 0, 0, 0, 0,  former_emp])
    if (aspect == 'promotion'):
        csv_writer.writerow([0, 0, 0, sentiment, 0, 0, 0, 0, 0,  former_emp])
    if (aspect == 'supervision'):
        csv_writer.writerow([0, 0, 0, 0, sentiment, 0, 0, 0, 0,  former_emp])
    if (aspect == 'rewards'):
        csv_writer.writerow([0, 0, 0, 0, 0, sentiment, 0, 0, 0,  former_emp])
    if (aspect == 'coworkers'):
        csv_writer.writerow([0, 0, 0, 0, 0, 0, sentiment, 0, 0,  former_emp])
    if (aspect == 'communication'):
        csv_writer.writerow([0, 0, 0, 0, 0, 0, 0, sentiment, 0,  former_emp])
    if (aspect == 'benefits'):
        csv_writer.writerow([0, 0, 0, 0, 0, 0, 0, 0, sentiment,  former_emp])