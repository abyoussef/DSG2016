import csv
import cv2
import matplotlib.pyplot as plt
import pandas

plt.style.use('ggplot')

with open('id_train.csv', 'rb') as csvfile:
	data = csv.DictReader(csvfile)

	cont_label = {1 : 0, 2 : 0, 3 : 0, 4 : 0}
	hlist = {}
	wlist = {}

	for row in data:
		id = row['Id']
		label = int(row['label'])
		img = cv2.imread('roof_images/' + id + '.jpg')

		if img.shape in cont_size:
			cont_size[img.shape] += 1
		else:
			cont_size[img.shape] = 1
		cont_label[label] += 1

		h, w = img.shape[0] / 10, img.shape[1] / 10
		if h in hlist:
			hlist[h] += 1
		else:
			hlist[h] = 1

		if w in wlist:
			wlist[w] += 1
		else:
			wlist[w] = 1

	print cont_label
	pandas.Series(hlist).plot(kind='bar', title='height')
	plt.show()
	pandas.Series(wlist).plot(kind='bar', title='width')
	plt.show()
