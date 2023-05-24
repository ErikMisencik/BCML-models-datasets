import json
import string
import re

from langdetect import detect

all_texts = []
reformed_texts = []

# Load the JSON data from the file
with open('ChangeMyMind_dataset_texts_unclean.json', encoding='utf-8') as f:
    data = json.load(f)

    with open("ChangeMyMind_dataset_texts_clean.json", "w", encoding='ascii', errors='ignore') as k:
        k.write('[\n')
        for i, item in enumerate(data):
            # Extract the text1 and text2 from the first item in the JSON data
            text = item[1]

            # Changes/Cleaning to top text
            text = text.replace('\n', ' ')
            text = text.replace('"', '').replace('|', ' ').replace('\\', ' ')
            text = text.rstrip()
            text = text.lstrip('-_ ')
            # Replace non-ASCII characters and emoticons with whitespace
            text = ''.join([' ' if ord(c) > 127 or c in string.punctuation else c for c in text])
            text = text.rstrip(string.punctuation)
            text = " ".join(text.split())

            # If top and bottom strings are empty, skip this file
            if not text.strip():
                continue

            if len(text) >= 100:
                continue

            if not re.search(r'\d', text):
                lang = detect(text)
                if lang != 'en':
                    continue

            text = text.lower()

            # Write the reformatted meme text to file
            k.write(f'["123456", "{text}|"],\n')
            all_texts.append(text)

    # Remove duplicate meme texts
    for i in range(len(all_texts)):
        is_unique = True
        for j in range(i + 1, len(all_texts)):
            if all_texts[i] == all_texts[j]:
                is_unique = False
                break
        if is_unique:
            reformed_texts.append(all_texts[i])

    # Write the reformatted meme texts to a new file
    with open("ChangeMyMind_dataset_texts_clean.json", "w", encoding='ascii', errors='ignore') as t:
        t.write('[\n')
        for i, item in enumerate(reformed_texts):
            if i == len(reformed_texts) - 1:
                t.write(f'["123456", "{item}|"]\n')
            else:
                t.write(f'["123456", "{item}|"],\n')
        t.write(']')
        print("done")