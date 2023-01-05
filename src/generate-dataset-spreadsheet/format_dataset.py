import os
import csv

excluded_reviews = open('excluded_reviews.csv', 'a', encoding="utf8")
e_writer = csv.writer(excluded_reviews, delimiter=',', lineterminator='\n')
e_writer.writerow(["job", "avaliation" ,"former_employee", "excluded_reason"])

included_reviews = open('final_dataset.csv', 'a', encoding="utf8")
i_writer = csv.writer(included_reviews, delimiter=',', lineterminator='\n')
i_writer.writerow(["job", "pros", "cons", "advice" ,"former_employee"])

def verify_valid_job_title(job_title):
    if(job_title == ''):
        return False
    return True

def create_invalid_reviews_dataset(job, avaliation, former, reason):
    e_writer.writerow([job, avaliation, former, reason])

def create_valid_reviews_dataset(job, pros, cons, advice, former):
    i_writer.writerow([job, pros, cons, advice, former])


def format_employee_avaliation(avaliation):
    av = avaliation.split("Contras:")
    pros = av[0]
    pros = pros.replace('Prós:', '')
    cons = av[1]
    if ('Conselhos para presidência:' in cons):
        temp = cons.split('Conselhos para presidência:')
        cons1 = temp[0]
        advice = temp[1]
        return pros, cons1, advice
    return pros, cons, 'NA'
    

def format_employee_job_title(job_title):
    if('Ex-funcionário' in job_title):
        temp = job_title.split(' - ')
        new_job_title = temp[0]
        return new_job_title
    else:
        return job_title

def verify_valid_employee_avaliation(avaliation):
    if (avaliation == ''):
        return False
    if ('loren ipsum' in avaliation.lower()):
        return False
    return True

def verify_if_it_employee(job_title):
    it_jobs = ['sistema', 'desenvolvedor', 'engenheiro de software', 'java', 'banco de dados', 'python', '.net', 'php', 'suporte', 'programador', 'líder de projeto', 'projeto', 'gerente de projeto', 'testador']
    for job in it_jobs:
        if job in job_title.lower():
            return True

with open('reviews_without_dupes.csv','r', encoding="utf8") as in_file:
    dataset_reader = csv.DictReader(in_file, delimiter=",")

    #limpa os arquivos csv antes da escrita dos dados
    excluded_reviews.seek(0)
    excluded_reviews.truncate()

    included_reviews.seek(0)
    included_reviews.truncate()

    invalid_job_title = 0
    non_it_job = 0
    invalid_avaliation = 0
    valid_responses = 0

    for line in dataset_reader:
        job_title = format_employee_job_title(line['user'])
        if(verify_valid_job_title(job_title)):
            if (verify_valid_employee_avaliation(line['avaliation'])):
                if (verify_if_it_employee(job_title)):
                    pros, cons, advice = format_employee_avaliation(line['avaliation'])
                    valid_responses += 1
                    if (line['former_employeer'] == "False"):
                        create_valid_reviews_dataset(job_title, pros, cons, advice, 0)
                    else:
                        create_valid_reviews_dataset(job_title, pros, cons, advice, 1)
                else:
                    non_it_job += 1
                    if(line["former_employeer"]  == "False"):
                        create_invalid_reviews_dataset(line["user"], line["avaliation"], 0, "Função não relacionada a área de TI")
                    else:
                        create_invalid_reviews_dataset(line["user"], line["avaliation"], 1, "Função não relacionada a área de TI")
            else:
                create_invalid_reviews_dataset(line["user"], line["avaliation"], 1, "Avaliação Inválida")
                invalid_avaliation += 1
        else:
            invalid_job_title += 1
            if(line["former_employeer"] == "False"):
                create_invalid_reviews_dataset(line["user"], line["avaliation"], 0, "Função não especificada")
            else:
                create_invalid_reviews_dataset(line["user"], line["avaliation"], 1, "Função não especificada")

print("Registros excluídos por não possuir função do reviewer: " +str(invalid_job_title))
print("Registros excluídos por não possuir avaliação válida: " +str(invalid_avaliation))
print("Registros excluídos por não serem avaliações de profissionais de TI: " +str(non_it_job))
print("Registros válidos: " +str(valid_responses))
excluded_reviews.close()
included_reviews.close()