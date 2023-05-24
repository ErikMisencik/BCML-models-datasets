import json

# Open the input JSON file for reading
# with open('drake_dataset_texts_unclean.json', 'r', encoding='utf-8') as input_file:
with open('ChangeMyMind_dataset_texts_unclean.json', 'r', encoding='utf-8') as input_file:
    # Load the JSON data as a list of rows
    rows = json.load(input_file)

# Add a '|' character to the end of the second column for each row
for row in rows:
    row[1] += '|'

# Open the output JSON file for writing
# with open('drake_dataset_texts_unclean.json', 'w', encoding='utf-8') as output_file:
with open('ChangeMyMind_dataset_texts_unclean.json', 'w', encoding='utf-8') as output_file:
    # Write the modified rows to the output JSON file
    output_file.write('[\n')
    for i, row in enumerate(rows):
        if i > 0:
            output_file.write(',\n')
        json.dump(row, output_file, ensure_ascii=False, separators=(',', ':'))
    output_file.write('\n]')