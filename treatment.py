# -*- coding: utf-8 -*-

import csv
import numpy as np

# Opening the original file and reading data into an array
with open("palio_features.csv", "r") as f:
	next(f)
	reader = csv.reader(f)
	my_list = list(reader)

f.close()

features = np.array(my_list)

# Cleaning up data, ending up only with desirable features
features_7 = []
features_8 = []
features_9 = []
features_cleaned = []

for feature in features:
	if len(feature) == 7:
		features_7.append(feature)
	if len(feature) == 8:
		features_8.append(feature)
	if  len(feature) == 9:
		features_9.append(feature)

for feature in features_7:
	if (feature[3].find(" ") == -1 and feature[3].find("ADVENTURE") == -1):
		features_cleaned.append([feature[0].replace(".", ""),feature[3],feature[4].replace(" km", "").replace(".",""),feature[5]])
for feature in features_8:
	if (feature[4].find(" ") == -1):
		features_cleaned.append([feature[0].replace(".", ""),feature[4],feature[5].replace(" km", ""),feature[6]])
for feature in features_9:
	if (feature[4].find(" ") == -1):
		features_cleaned.append([feature[0].replace(".", ""),feature[4],feature[5].replace(" km", ""),feature[6]])

features = np.array(features_cleaned)

# Normalizing functions
def Standard(y):
	mean = np.mean(y)
	std = np.std(y)
	y = (y - mean) / std
	return y

def MaxMin(y):
	return (y - np.min(y)) / (np.max(y) + np.min(y))

# Converting Doors feature into binary values
for feature in features:
	if (feature[3] == '4' or feature[3] == '04'): feature[3] = 1
	if (feature[3] == '5' or feature[3] == '05'): feature[3] = 1
	if (feature[3] == '2' or feature[3] == '02'): feature[3] = -1
	if (feature[3] == '3' or feature[3] == '03'): feature[3] = -1

# Normalizing price, km and age
price = features[:,0].astype("int")
features[:,0] = MaxMin(price)

age = features[:,1].astype("int")
features[:,1] = Standard(age)

km = features[:,2].astype("int")
features[:,2] = Standard(km)

# Exporting treated, normalized data
with open('normalized_palio_features.csv', 'w') as f:
	writer = csv.writer(f)
	for values in features:
		writer.writerow(values)
f.close()