#!/usr/bin/python
from lxml import html
import requests
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# get the leaderboard page
page = requests.get("https://www.kaggle.com/c/sf-crime/leaderboard")
tree = html.fromstring(page.content)

# scores are encoded like this:
'''
<abbr class="score" title="0.00327">2.04753</abbr>
'''

# insert those if you want to visualize your scores
# my scores
sns.color_palette("Set2", 10)
values = [
        ["Decision Tree F1", 2.62],
        ["Decision Tree F2", 2.578],
        ["Decision Tree F3", 2.574],
        ["Decision Tree F4", 2.567],
        ["Decision Tree F5", 2.535],

        ["RandomForest F1", 2.586],
        ["RandomForest F2", 2.415],
        ["RandomForest F3", 2.415],
        ["RandomForst F4", 2.363],
        ["RandomForest F5", 2.344],

        ["Adaboost F1", 2.877],
        ["Adaboost F2", 2.871],
        ["Adaboost F3", 2.870],
        ["Adaboost F4", 2.86],
        ]

# calculate the best value of the above (here: smaller is better)
my_best_value = 10000.
for v in values:
    if v[1] < my_best_value:
        my_best_value = v[1]
print "My best value is %.2f" % my_best_value

colors = sns.color_palette("Set2", len(values))

# parse title and scores
title = tree.xpath('//title/text()')[0]
scores = tree.xpath('//abbr[@class="score"]/text()')
scores = np.array(scores).astype('float32')

# calculate some stats before cutting off
mean = np.mean(scores)
median = np.median(scores)

# for each value, calculate the top percent of the leaderboard we are in
for v in values:
    num_before = scores[scores < v[1]].shape[0]
    best_percent = ((num_before + 1.) / float(scores.shape[0])) * 100.
    print "Value for %s is in the best %.0f percent!" % (v[0], best_percent)


# calculate how many before my best score
# and in which of the top percentages we scored
num_before = scores[scores < my_best_value].shape[0]
best_percent = ((num_before + 1.) / float(scores.shape[0])) * 100.
print "Your best score is in the top %.0f percent!" % best_percent 

# for the sake of displaying, 
# cut off scores that are > cutoff, put in last bin
cutoff = 4.
scores[scores > 4] = 4.

# plot the histogram of the scores
fig = plt.figure()
plt.suptitle("Histogram for %s" % title)
plt.title("%d entries, mean: %.1f, median: %.1f\nBest score in the top %.0f percent" % (scores.shape[0], mean, median, best_percent))
binsize = .1
plt.hist(scores, bins=np.arange(round(min(scores),1),max(scores)+binsize/2., binsize))

# set xticks and labels
xticks = np.arange(round(min(scores),1), max(scores)+binsize/2., binsize)
labels = ["%.1f"%x for x in xticks]
labels[-1] = ">%.1f" % cutoff
plt.xticks(xticks, labels)

# plot my own scores onto the histogram
for i,v in enumerate(values):
    plt.axvline(x=v[1], c=colors[i], label=v[0])
plt.legend()

plt.show()
