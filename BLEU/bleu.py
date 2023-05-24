import pandas as pd
import nltk
from nltk.translate.bleu_score import sentence_bleu
import csv

infile_path = 'imgflip_ai_meme_captions_drake.csv'  # vstupny csv subor kde je text ktory chces ohodnotit (1 riadok jeden inzerat/meme..)
listings_generated = []
reference_dataset = []

# otvori subor z ktoreho precita data ktore chces ohodnotit
with open(infile_path, 'r') as infile:
    reader = csv.reader(infile)
    for line in reader:
        listings_generated.append(line[0].lower())  # Convert to lowercase and append

# otvori csv subor kde mas original data ku ktorym to budes porovnavat, tiez csv 1 riadok 1 inzerat/meme...
with open('real_meme_captions_drake.csv', 'r') as infile:
    reader = csv.reader(infile)
    for line in reader:
        reference_dataset.append(nltk.word_tokenize(line[0].lower()))

reference_dataset = reference_dataset[:20325]

# samotne pocitanie BLEU
scores = []
bleu_file = open('bleu_AI2.csv', 'w', newline='')  # vytvori subor kde zapise skore
writer = csv.writer(bleu_file)

for idx, listing in enumerate(listings_generated):
    listing_words = nltk.word_tokenize(listing)

    # tu sa pocita uz skore, weights su vahy ngramov na pouzitie
    # napr 0, 0.33, 0.33, 0.33 je pocitanie s 2,3,4 ngramami
    bleu_score = sentence_bleu(reference_dataset, listing_words, weights=(0.3, 0.3, 0.3))
    writer.writerow([bleu_score])
    print(bleu_score)
bleu_file.close()


# import pandas as pd
# import nltk
# from nltk.translate.bleu_score import sentence_bleu
# import csv
#
# infile_path = 'ericek_ai_meme_captions_drake.csv'  # vstupny csv subor kde je text ktory chces ohodnotit (1 riadok jeden inzerat/meme..)
# listings_generated = []
# reference_dataset = []
#
# # otvori subor z ktoreho precita data ktore chces ohodnotit
# with open(infile_path, 'r') as infile:
#     reader = csv.reader(infile)
#     for line in reader:
#         listings_generated.append(line[0].lower())  # Convert to lowercase and append
#
# # otvori csv subor kde mas original data ku ktorym to budes porovnavat, tiez csv 1 riadok 1 inzerat/meme...
# with open('real_meme_captions_drake.csv', 'r') as infile:
#     reader = csv.reader(infile)
#     for line in reader:
#         reference_dataset.append(list(line[0].lower()))
#
# reference_dataset = reference_dataset[:20325]
#
# # samotne pocitanie BLEU
# scores = []
# bleu_file = open('bleu_234g.csv', 'w', newline='')  # vytvori subor kde zapise skore
# writer = csv.writer(bleu_file)
#
# for idx, listing in enumerate(listings_generated):
#     listing_chars = list(listing)
#
#     # tu sa pocita uz skore, weights su vahy ngramov na pouzitie
#     # napr 0, 0.33, 0.33, 0.33 je pocitanie s 2,3,4 ngramami
#     bleu_score = sentence_bleu(reference_dataset, listing_chars, weights=(1/3, 1/3, 1/3))
#     writer.writerow([bleu_score])
#     print(bleu_score)
# bleu_file.close()