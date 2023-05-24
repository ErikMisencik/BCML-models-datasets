import statistics

import matplotlib.pyplot as plt
import csv

bleu_scores = []

# nacitanie BLEU skore
with open('bleu_AI2.csv', 'r') as score_file:
    reader = csv.reader(score_file)
    for score in reader:
        bleu_scores.append(round(float(score[0]), 2))

print(bleu_scores)
# Create a histogram
plt.hist(bleu_scores, bins=10, edgecolor='black', linewidth=1)
plt.xlabel('BLEU skóre')
plt.yticks(range(0, 55, 5))
plt.ylabel('Počet vtipnych popisov')
plt.title('Histogram BLEU skóre vygenerovaných popisov')

mean = statistics.mean(bleu_scores)
plt.axvline(mean, color='red', linestyle='dashed', linewidth=2)
plt.text(mean-0.12, plt.ylim()[1]*0.9, f'Priemer: {mean:.2f}', ha='center', va='top')

plt.savefig('HistrogramBleuAIV2.png')
plt.show()
