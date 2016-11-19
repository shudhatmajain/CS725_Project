import pandas
import numpy as np

vitals = pandas.read_csv("train/id_time_vitals_train.csv")
labs =  pandas.read_csv("train/id_time_labs_train.csv").drop("ID", axis=1).drop("TIME", axis=1)
# vitals = pandas.read_csv("train/short_vitals.csv")
# labs =  pandas.read_csv("train/short_labs.csv").drop("ID", axis=1).drop("TIME", axis=1)
ages =  pandas.read_csv("train/id_age_train.csv")
labels = pandas.read_csv("train/id_label_train.csv")

agesDict = dict()
labelDict = dict()

for r in ages.iterrows():
	row = r[1]
	agesDict[row[0]]=row[1]

for r in labels.iterrows():
	row = r[1]
	labelDict[row[0]]=row[1]

vitals = vitals.as_matrix()
labs = labs.as_matrix()

data = np.concatenate((vitals, labs), axis=1)

means = np.nanmean(data, axis=0)

newdata = []
for row in data:
	d = list(row)
	d.append(agesDict[row[0]])
	d.append(labelDict[row[0]])
	newdata.append(d)
data=newdata

nPeople = 3594
blanklabel = "nan"

peopleSeries = []
cur = []
prev = 1
cleandata = []
# deaths = 0

for row in data:
	if row[0] == prev:
		cur.append(row)
		continue
	# fillData
	for currow in cur:
		# deaths += currow[-1]
		currow[-1] *= np.exp(-(cur[-1][1]-currow[1])/2000)
		for i in range(len(currow)):
			if np.isnan(currow[i]):
				currow[i] = means[i]
		times = 1
		if currow[-1]>0:
			times = 5
		for _ in range(times):
			cleandata.append(currow)
	# peopleSeries.append(cur)
	prev = row[0]
	cur = [row]
for currow in cur:
	# deaths += currow[-1]
	currow[-1] *= np.exp(-(cur[-1][1]-currow[1])/2000)
	for i in range(len(currow)):
		if np.isnan(currow[i]):
			currow[i] = means[i]
	times = 1
	if currow[-1] > 0:
		times=5
	for _ in range(times):
		cleandata.append(currow)
# peopleSeries.append(cur)

data = np.array(data)
cleandata = np.array(cleandata)

np.save('data2',cleandata)