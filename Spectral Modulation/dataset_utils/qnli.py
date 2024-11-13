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

# Function to load, process, and save dataset samples
def get_dataset():
    dataset = load_dataset("SetFit/qnli")  # Adjust path or use the dataset name as appropriate
    paper_train = list(dataset["train"])
    paper_dev = list(dataset["validation"])
    all_samples = paper_train + paper_dev
    all_samples = random.sample(all_samples, 4000)  # Randomly sample 4000 entries
    
    text1 = [dp["text1"] for dp in all_samples]
    text2 = [dp["text2"] for dp in all_samples]
    label_text = [dp["label_text"] for dp in all_samples]
    
    assert len(all_samples) == 4000
    assert len(text1) == len(text2) == len(label_text) == 4000
    
    # Create prompts and adjust label values
    prompt = []
    for i in range(len(all_samples)):
        text1[i] = text1[i].strip().replace("\n", "").strip()
        text2[i] = text2[i].strip().replace("\n", "").strip()
        if label_text[i] != "entailment":
            label_text[i] = "non-entailment"
        prompt.append(
            f'Identify the relation between the following premises and hypotheses, choosing from the options \'entailment\' or \'non-entailment\'. Premise: "{text2[i]}". Hypothesis: "{text1[i]}". Relation:'
        )
    
    # Save prompts and labels to CSV
    lists_to_csv(prompt, label_text, 'qnli.csv')  # Modify file path as needed
    return prompt, label_text

