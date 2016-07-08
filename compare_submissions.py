import csv
import cv2

submission_name1 = 'submission1'
submission_name2 = 'submission2'
detailed = True
classes = 4
limit = 10

if detailed:
	file1 = submission_name1 + '_detailed.csv'
	file2 = submission_name2 + '_detailed.csv'
else:
	file1 = submission_name1 + '.csv'
	file2 = submission_name2 + '.csv'

with open(file1, 'rb') as fsub1:
	with open(file2, 'rb') as fsub2:
		sub1 = csv.DictReader(fsub1)
		sub2 = csv.DictReader(fsub2)
		data1 = []
		data2 = []
		for row in sub1:
			data1.append(row)
		for row in sub2:
			data2.append(row)

		assert(len(data1) == len(data2))
		len = len(data1)
        comparison_matrix = []
        for i in range(0,classes):
            comparison_matrix.append([0] * classes)
        cont_discrepancies = 0

        for i in range(0,len):
            cur1, cur2 = data1[i], data2[i]
            assert(cur1['Id'] == cur2['Id'])
            id = cur1['Id']

            comparison_matrix[ int(cur1['label']) - 1 ][ int(cur2['label']) - 1 ] += 1

            if cur1['label'] != cur2['label']:
                cont_discrepancies += 1
                if limit > 0:
                    print 'Id = ', cur1['Id'] ,'sub1 =', cur1['label'], 'sub2 =', cur2['label']
                    i = cv2.imread('roof_images/' + id + '.jpg')
                    cv2.imshow('img', i)
                    cv2.waitKey(0)
                    limit -= 1
                    if detailed:
                        print cur1['Id']
                        print cur1['label'], "(" + cur1['cat' + cur1['label']] +  ")", \
                                cur1['cat1'], cur1['cat2'], cur1['cat3'], cur1['cat4']
                        print cur2['label'], "(" + cur2['cat' + cur2['label']] +  ")", \
                                cur2['cat1'], cur2['cat2'], cur2['cat3'], cur2['cat4']

        print cont_discrepancies
        for i in range(0,classes):
            print(comparison_matrix[i])
