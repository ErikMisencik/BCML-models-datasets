import json
import string
import re

from langdetect import detect

all_texts = []
reformed_texts = []

# Load the JSON data from the file
with open('BatmanSlapRobin_dataset_texts_unclean.json', encoding='utf-8') as f:
    data = json.load(f)

    with open("BatmanSlapRobin_dataset_texts_clean.json", "w", encoding='ascii', errors='ignore') as k:
        k.write('[\n')
        for i, item in enumerate(data):
            # Extract the text1 and text2 from the first item in the JSON data
            text = item[1]
            text_parts = text.split('|', 1)
            top = text_parts[0].strip()

            if len(text_parts) > 1:
                # Check if there is a second '|' in the text
                if '|' in text_parts[1]:
                    bottom_parts = text_parts[1].split('|', 1)
                    bottom = bottom_parts[0].strip()
                else:
                    bottom = text_parts[1].strip()
            else:
                bottom = ''

            # Changes/Cleaning to top text
            top = top.replace('\n', ' ')
            top = top.replace('"', '').replace('|', ' ').replace('\\', ' ')
            top = top.rstrip()
            top = top.lstrip('-_ ')
            # Replace non-ASCII characters and emoticons with whitespace
            top = ''.join([' ' if ord(c) > 127 or c in string.punctuation else c for c in top])
            top = top.rstrip(string.punctuation)
            top = " ".join(top.split())

            # Changes/Cleaning to bottom text
            bottom = bottom.replace('\n', ' ')
            bottom = bottom.replace('"', '').replace('|', ' ').replace('\\', ' ')
            bottom = bottom.rstrip()
            bottom = bottom.lstrip('-_ ')
            # Replace non-ASCII characters and emoticons with whitespace
            bottom = ''.join([' ' if ord(c) > 127 or c in string.punctuation else c for c in bottom])
            bottom = bottom.rstrip(string.punctuation)
            bottom = " ".join(bottom.split())

            # If top and bottom strings are empty, skip this file
            if not top.strip() or not bottom.strip():
                continue

            top = top.lower()
            bottom = bottom.lower()

            meme_text = top + '|' + bottom

            if len(meme_text) >= 100:
                continue

            if not re.search(r'\d', meme_text):
                lang = detect(meme_text)
                if lang != 'en':
                    continue

            # Write the reformatted meme text to file
            k.write(f'["333444", "{top}|{bottom}|"],\n')
            all_texts.append(meme_text)

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
    with open("BatmanSlapRobin_dataset_texts_clean.json", "w", encoding='ascii', errors='ignore') as t:
        t.write('[\n')
        for i, item in enumerate(reformed_texts):
            if i == len(reformed_texts) - 1:
                t.write(f'["333444", "{item}|"]\n')
            else:
                t.write(f'["333444", "{item}|"],\n')
        t.write(']')
        print("done")