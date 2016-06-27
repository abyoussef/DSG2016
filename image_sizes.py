import csv
import cv2
import matplotlib.pyplot as plt
import pandas

plt.style.use('ggplot')

with open('id_train.csv', 'rb') as csvfile:
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

		h, w = img.shape[0] / 10, img.shape[1] / 10
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
	plt.scatter(hlist, wlist)
	plt.show()
	pandas.Series(hcont).plot(kind='bar', title='height')
	plt.show()
	pandas.Series(wcont).plot(kind='bar', title='width')
	plt.show()
