## Label matching code

from sentence_transformers import SentenceTransformer, util
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score, jaccard_score
from sklearn.preprocessing import MultiLabelBinarizer

# Load the pre-trained sentence transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

descriptions = {
    'Unnecessary': 'The tweet indicates vaccines are unnecessary, or that alternate cures are better.',
    'Mandatory': 'Against mandatory vaccination — The tweet suggests that vaccines should not be made mandatory.',
    'Pharma': 'Against Big Pharma — The tweet indicates that the Big Pharmaceutical companies are just trying to earn money, or the tweet is against such companies in general because of their history.',
    'Conspiracy': 'Deeper Conspiracy — The tweet suggests some deeper conspiracy, and not just that the Big Pharma want to make money (e.g., vaccines are being used to track people, COVID is a hoax)',
    'Political': 'Political side of vaccines — The tweet expresses concerns that the governments / politicians are pushing their own agenda though the vaccines.',
    'Country': 'Country of origin — The tweet is against some vaccine because of the country where it was developed / manufactured',
    'Rushed': 'Untested / Rushed Process — The tweet expresses concerns that the vaccines have not been tested properly or that the published data is not accurate.',
    'Ingredients': 'Vaccine Ingredients / technology — The tweet expresses concerns about the ingredients present in the vaccines (eg. fetal cells, chemicals) or the technology used (e.g., mRNA vaccines can change your DNA)',
    'Side-effect': 'Side Effects / Deaths — The tweet expresses concerns about the side effects of the vaccines, including deaths caused.',
    'Ineffective': 'Vaccine is ineffective — The tweet expresses concerns that the vaccines are not effective enough and are useless.',
    'Religious': 'Religious Reasons — The tweet is against vaccines because of religious reasons',
    'None': 'No specific reason stated in the tweet, or some reason other than the given ones.'
}

# Generate sentence embeddings for the values in the 'questions' dictionary
descriptions_embeddings = {key: model.encode(value, convert_to_tensor=True, show_progress_bar=False) for key, value in descriptions.items()}

# Initialize a dictionary to store the top matching labels for each sentence
top_matching_labels = {}

list_labels = []
k = 0
# Read sentences from a text file (replace 'your_text_file.txt' with the actual file path)
with open('./lora_prediction_flan_t5_base.txt', 'r') as file:
    current_label = None
    for line in file:
        k = k + 1;
        if(k % 1000 == 0):
            print(f"{k}th iteration")
        line = line.strip()
        temp_list = []
        # Check if "Pred:" appears in the line
        if "Pred:" in line:
            # Split the line at "Pred:" and take the part after it
            current_text = line.split("Pred:", 1)[1].strip()

            # Set the current label
            current_label = current_text
            # Split the line into sentences using full stops (periods) as separators
            sentences = current_label.split('.')
            sentences = [sentence.strip() for sentence in sentences if sentence.strip()]

            # Find the top matching key for each sentence
            for i, sentence in enumerate(sentences):
                words = sentence.split()

                # Define a set of articles and the word 'no'
                articles_and_no = set(['a', 'an', 'the', 'no'])

                # Check if all words in the sentence are in the set of articles and 'no'
                if all(word.lower() in articles_and_no for word in words):
                    continue

                similarities = {}
                embedding = model.encode(sentence, convert_to_tensor=True, show_progress_bar=False)
                for key, category_embedding in descriptions_embeddings.items():
                    cos_sim = util.pytorch_cos_sim(embedding, category_embedding)
                    similarities[key] = np.mean(cos_sim.cpu().numpy())

                top_matching_key = max(similarities, key=similarities.get)

                temp_list.append(top_matching_key.lower())
        list_labels.append(list(set(temp_list)))

flattened_data = [' '.join(map(str, sublist)) for sublist in list_labels]

# Convert it to a pandas DataFrame with a single column
df = pd.DataFrame(flattened_data, columns=['Combined_Column'])

# Join the elements within each sublist with space separation
df['Combined_Column'] = df['Combined_Column'].apply(lambda x: ''.join(x))

val_aiso = pd.read_csv('test_final_only_train.csv')
val_aiso['Predicted'] = df['Combined_Column']
#val_aiso.to_csv('predicted_test_flan_t5_base.csv')


ground_truth_labels = val_aiso['labels'].str.split()  # Assuming labels are separated by spaces
predicted_labels = val_aiso['Predicted'].str.split()        # Assuming labels are separated by spaces

#print(predicted_labels) #Initialize the MultiLabelBinarizer to convert labels into binary format
mlb = MultiLabelBinarizer()


# Transform the ground truth and predicted labels into binary format
ground_truth_binary = mlb.fit_transform(ground_truth_labels)
predicted_binary = mlb.transform(predicted_labels)



# Calculate the F1 macro score
f1_macro = f1_score(ground_truth_binary, predicted_binary, average='macro')

# Print the F1 macro score
print("F1 Macro Score:", f1_macro)


f1_micro = f1_score(ground_truth_binary, predicted_binary, average='micro')


print("F1 micro score", f1_micro)

accuracy_score = accuracy_score(ground_truth_binary, predicted_binary)
print(accuracy_score)

weight = f1_score(ground_truth_binary, predicted_binary, average="weighted", zero_division=0)
jacc = jaccard_score(ground_truth_binary, predicted_binary, average="samples", zero_division=0)

print('weighted', weight, 'jacc', jacc)
