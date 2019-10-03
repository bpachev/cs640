from numpy.linalg import eigh
import numpy as np
from argparse import ArgumentParser, FileType
from numpy.random import permutation
from sklearn.cluster import MiniBatchKMeans
"""
This file is a script to compute clusters for vector quantitization.
The current algorithm is K-Means clustering.

"""


def compute_train_indices(lengths, train_mask):
	num_training = np.sum(lengths[train_mask])
	indices = np.zeros(num_training, dtype=np.int64)
	start_index = 0
	start_train = 0
	for i in xrange(lengths.size):
		if train_mask[i]:
			indices[start_train:start_train+lengths[i]] = np.arange(start_index, start_index+lengths[i])
			start_train += lengths[i]

		start_index += lengths[i]
	return indices

if __name__ == "__main__":
	parser = ArgumentParser()
	parser.add_argument("infile",type=FileType("r"))
	parser.add_argument("--outfile", type=FileType("w"))
	parser.add_argument("--num-samples", type=int, default=50000)
	parser.add_argument("--num-clusters", type=int, default=1000)
	parser.add_argument("--mahal",action="store_true")
	args = parser.parse_args()
	
	data = np.load(args.infile)
	inds = compute_train_indices(data['lengths'], data['train_mask'])
	sample_inds = permutation(inds)[:args.num_samples]
	sample_vecs = data['features'][sample_inds]

	if args.mahal:
		cov = np.cov(sample_vecs.T)
		eigs, V = eigh(cov)
		transform_mat = (V*eigs**(-.5)).dot(V.T)
		sample_vecs = sample_vecs.dot(transform_mat)
	
	clusterer = MiniBatchKMeans(n_clusters=args.num_clusters, batch_size=int(.15*args.num_samples))
	clusterer.fit(sample_vecs)
	print clusterer.inertia_
	np.savez(args.outfile, cluster_centers=clusterer.cluster_centers_, labels=clusterer.labels_, sample_inds=sample_inds, transform_mat=transform_mat)
	import matplotlib.pyplot as plt
	plt.hist(clusterer.labels_, bins=args.num_clusters)
	plt.show()
#	print data['features'].shape, data['lengths'].shape, data['files'].shape
	

