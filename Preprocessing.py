import re
from collections import defaultdict
import pandas as pd
import os
import QuechuaDataset

language_folder = 'dialects_mixed_txt'

# Define your directory paths correctly
directory_paths = {
    'qub': language_folder + '/qub',
    'quf': language_folder + '/quf',
    'quh': language_folder + '/quh',
    'quk': language_folder + '/quk',
    'qul': language_folder + '/qul',
    'qup': language_folder + '/qup',
    'quw': language_folder + '/quw',
    'qux': language_folder + '/qux',
    'quy': language_folder + '/quy',
    'quz': language_folder + '/quz',
    'qvc': language_folder + '/qvc',
    'qve': language_folder + '/qve',
    'qvi': language_folder + '/qvi',
    'qvm': language_folder + '/qvm',
    'qvn': language_folder + '/qvn',
    'qvo': language_folder + '/qvo',
    'qvw': language_folder + '/qvw',
    'qvz': language_folder + '/qvz',
    'qwh': language_folder + '/qwh',
    'qxl': language_folder + '/qxl',
    'qxh': language_folder + '/qxh',
    'qxn': language_folder + '/qxn',
    'qxo': language_folder + '/qxo',
    'qxr': language_folder + '/qxr'
}

dialect2index = {
    'qub': 0,
    'quf': 1,
    'quh': 2,
    'quk': 3,
    'qul': 4,
    'qup': 5,
    'quw': 6,
    'qux': 7,
    'quy': 8,
    'quz': 9,
    'qvc': 10,
    'qve': 11,
    'qvi': 12,
    'qvm': 13,
    'qvn': 14,
    'qvo': 15,
    'qvw': 16,
    'qvz': 17,
    'qwh': 18,
    'qxl': 19,
    'qxh': 20,
    'qxn': 21,
    'qxo': 22,
    'qxr': 23,
}

node2index = {
    "quechua1": 0,
    "quechua2": 1,
    "cajamarca-lambayeque": 2,
    "columbia-ecuador": 3,
    "san-martin": 4,
    "ap-am-ah": 5,
    "huaylay": 6,
    "panao": 7,
    "conchucos": 8,
    "bolivian-argentinian": 9,
    "cuscan": 10,
    "ecuadorian-quechua-b": 11,
    "imbabura-columbia-oriente": 12,
    "columbia-oriente": 13,
    'qub': 14,
    'quf': 15,
    'quh': 16,
    'quk': 17,
    'qul': 18,
    'qup': 19,
    'quw': 20,
    'qux': 21,
    'quy': 22,
    'quz': 23,
    'qvc': 24,
    'qve': 25,
    'qvi': 26,
    'qvm': 27,
    'qvn': 28,
    'qvo': 29,
    'qvw': 30,
    'qvz': 31,
    'qwh': 32,
    'qxl': 33,
    'qxh': 34,
    'qxn': 35,
    'qxo': 36,
    'qxr': 37,
}

hierarchy = {
    "quechua1": ["ap-am-ah", "huaylay", "qvn", "qvw"],
    "huaylay": ["qvh", "qwh", "conchucos"],
    "conchucos": ["qxn", "qxo"],
    "ap-am-ah": ["qub", "panao"],
    "panao": ["qvm", "qxh"],
    "quechua2": ["quy", "bolivian-argentinian", "cuscan"],
    "bolivian-argentinian": ["quh", "qul"],
    "cuscan": ["quz", "qve"],
    "cajamarca-lambayeque": ["qvc", "quf"],
    "columbia-ecuador": ["qxl", "ecuadorian-quechua-b"],
    "ecuadorian-quechua-b": ["qxr", "imbabura-columbia-oriente"],
    "imbabura-columbia-oriente": ["qvi", "columbia-oriente"],
    "columbia-oriente": ["inb", "qup", "quw", "qvo", "qvz"],
}

def split_document(text, max_length=250, overlap=50):
    # Split text into words
    words = text.split()
    parts = []
    if len(words) <= max_length:
        return [text]  # Return the entire text if it's short enough

    i = 0
    while i < len(words):
        # Ensure that we don't exceed the text length
        end_index = min(i + max_length, len(words))
        # Join the selected range of words back into a string
        chunk_text = " ".join(words[i:end_index])
        parts.append(chunk_text)
        i += (max_length - overlap)

    return parts

def remove_numbers(text):
    # Remove numbers using regular expression
    text_without_numbers = re.sub(r'\d+', '', text)
    return text_without_numbers

def createChunkedDf():
    dialect_data_df = pd.DataFrame(columns=['dialect', 'file', 'chunk', 'text'])
    
    for dialect in directory_paths.keys():
        filePaths = list(filter(lambda f: f.endswith(".txt"), list(map(lambda x: os.path.join(directory_paths[dialect], x), os.listdir(directory_paths[dialect])))))

    for filePath in filePaths:
        with open(filePath, "r", encoding='latin-1') as f:
            text = f.read().strip()

            text = remove_numbers(text)

            chunks = split_document(text)
            chunk_count = 0
            for c in chunks:
                dialect_data_df = pd.concat([dialect_data_df, pd.DataFrame({'dialect': [dialect], 'file': [filePath], 'chunk': [chunk_count], 'text': [c]})], ignore_index=True)
                # chunked_data_file.write(f"{dialect},{filePath},{chunk_count},{c}\n")
                chunk_count += 1
    
    dialect_data_df.to_csv('chunked_data.csv', index=False)

def get_hierarchy():
    child2parent = defaultdict(lambda: None)
    for parent, children in hierarchy.items():
        for child in children:
            child2parent[child] = parent

    return child2parent

def createDataset():
    child2parent = get_hierarchy()
    
    createChunkedDf()
    
    dataset = QuechuaDataset.QuechuaDataSet('chunked_data.csv', node2index, child2parent)

    print("Created dataset of length", len(dataset))


def main():
    createDataset()


if __name__ == "__main__":
    main()