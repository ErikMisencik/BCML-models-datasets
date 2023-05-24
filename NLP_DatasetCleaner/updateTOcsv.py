import json
import csv

# Open the JSON file
with open('drake_dataset_texts_clean.json') as f:
    data = json.load(f)

# Extract the second column of data
column2 = [row[1] for row in data]

# Open a CSV file for writing
with open('real_meme_captions_drake.csv', 'w', newline='') as f:
    writer = csv.writer(f)

    # Write each row of the second column to the CSV file
    for row in column2:
        writer.writerow([row])