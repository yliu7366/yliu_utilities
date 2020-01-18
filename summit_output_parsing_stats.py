#script to parse Summit output files for job statistics

import os, shutil
import natsort

root = 'stats_20200118'

outFiles = natsort.natsorted(os.listdir(root))

stats = dict()

print('Summit DDL test results 20200118\n')

print('Num. of Nodes\t', 'job id\t', 'Results')

success = 0
failure = 0
total = 0

for out in outFiles:
	fp = open(os.path.join(root, out))

	total += 1

	jobID = out[:-4]

	items = jobID.split('_')

	nodes = list()

	line = fp.readline()

	while line:
		if 'Signal: Segmentation fault (11)' in line:
			tokens = line.split()
			nodes.append(tokens[0][1:7])
		line = fp.readline()

	fp.close()

	if len(nodes) > 0:
		nodes = list(dict.fromkeys(nodes))
		stats[jobID] = nodes
		print('\t'+items[0][1:]+'\t', items[1]+'\t', 'Failure')
		failure += 1
	else:
		print('\t'+items[0][1:]+'\t', items[1]+'\t', 'Success')
		success += 1

print('\nTotal jobs:', total, 'Success:', success, 'Failure:', failure, '\n')

print('Num. of Nodes\t', 'job id\t', 'Segment fault on nodes')
for key in stats:
	items = key.split('_')
	print('\t'+items[0][1:]+'\t', items[1]+'\t', stats[key])

print('done.')
