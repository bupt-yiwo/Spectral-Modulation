import pandas as pd
import csv

class SST():
    def get_dataset(self, logger):
        df = pd.read_csv("SST.tsv",encoding= "utf-8",sep='\t', header=None)

        question = []
        answer = []

        for i in df.iloc[:,0]:
            question.append(i)
        for j in df.iloc[:,1]:
            answer.append(j)
                
       
        assert len(question) == len (answer)

        logger.log(f"Read dataset of size {len(answer)}.")

        return question,answer
    

def csv_to_lists(csv_filename):
    list1 = []
    list2 = []
    

    with open(csv_filename, mode='r') as file:
        reader = csv.reader(file)
        

        next(reader, None)
        
        for row in reader:
            if len(row) != 2:
                raise ValueError("Each row in the CSV file should contain two columns")
            list1.append(row[0])
            list2.append(row[1])
    
    return list1, list2
