import csv
import cv2
import matplotlib.pyplot as plt
import pandas

plt.style.use('ggplot')
files = ['id_train.csv', 'sample_submission4.csv']

for file in files:
	with open(file, 'rb') as csvfile:
		data = csv.DictReader(csvfile)

		cont_label = {1 : 0, 2 : 0, 3 : 0, 4 : 0}
		hcont = {}
		wcont = {}
		hlist = []
		wlist = []

		for row in data:
			id = row['Id']
			label = int(row['label'])
			img = cv2.imread('roof_images/' + id + '.jpg')

			cont_label[label] += 1

			h, w = img.shape[0], img.shape[1]
			hlist.append(h)
			wlist.append(w)
			if h in hcont:
				hcont[h] += 1
			else:
				hcont[h] = 1

			if w in wcont:
				wcont[w] += 1
			else:
				wcont[w] = 1

		print cont_label
		plt.scatter(hlist, wlist, label=file)
		plt.show()
		pandas.Series(hcont).plot(kind='bar', title=file + ' height')
		plt.show()
		pandas.Series(wcont).plot(kind='bar', title=file + ' width')
		plt.show()
