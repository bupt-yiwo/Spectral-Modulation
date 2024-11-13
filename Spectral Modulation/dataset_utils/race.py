from datasets import load_dataset
import random
import csv

# Set random seed for reproducibility
random.seed(0)

# Function to save two lists as a CSV file
def lists_to_csv(list1, list2, csv_filename):
    if len(list1) != len(list2):
        raise ValueError("The two lists must have the same length")
    
    with open(csv_filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Column1', 'Column2'])
        for item1, item2 in zip(list1, list2):
            writer.writerow([item1, item2])

# Function to read data from a CSV file into two lists
def csv_to_lists(csv_filename):
    list1, list2 = [], []
    with open(csv_filename, mode='r') as file:
        reader = csv.reader(file)
        next(reader, None)  # Skip the header row
        for row in reader:
            if len(row) != 2:
                raise ValueError("Each row in the CSV file must have exactly two columns")
            list1.append(row[0])
            list2.append(row[1])
    return list1, list2

# Helper function to convert letter answers to numbers
def convert_letter_to_number(letter):
    letter_to_number = {
        'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6
    }
    return letter_to_number.get(letter)

# Main function to process the dataset
def get_dataset():
    dataset = load_dataset("ehovy/race")  # Adjust path or use the dataset name as needed
    paper_train = list(dataset["train"])
    paper_dev = list(dataset["validation"])
    paper_test = list(dataset["test"])
    all_samples = paper_train + paper_dev + paper_test
    
    questions = [dp["question"] for dp in all_samples]
    passages = [dp["article"] for dp in all_samples]
    answers = [dp["answer"] for dp in all_samples]
    options = [dp["options"] for dp in all_samples]
    
    assert len(questions) == len(answers) == len(passages) == len(options)
    
    # Create prompts and labels
    prompt = []
    label = []
    for i in range(len(all_samples)):
        for j in range(4):
            is_correct = (convert_letter_to_number(answers[i]) == j)
            prompt.append(
                f'For the passage "{passages[i].strip()}", the question is "{questions[i].strip()}", is the answer "{options[i][j].strip()}" to this question right or wrong? The result is'
            )
            label.append(is_correct)
    
    # Save prompts and labels to CSV
    lists_to_csv(prompt, label, 'race.csv')  # Modify file path as needed
    return prompt, label

# Run the dataset processing function
get_dataset()

