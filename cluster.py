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
	parser.add_argument("--cluster", default="kmeans")
	args = parser.parse_args()
	
	if args.outfile is None: raise ValueError("Must specify outfile!")

	data = np.load(args.infile)
	inds = compute_train_indices(data['lengths'], data['train_mask'])
	sample_inds = permutation(inds)[:args.num_samples]
	sample_vecs = data['features'][sample_inds]
	
	transform_mat = None
	#Decorrelate the data, assuming constant variance
	if args.mahal:
		cov = np.cov(sample_vecs.T)
		eigs, V = eigh(cov)
		transform_mat = (V*eigs**(-.5)).dot(V.T)
		sample_vecs = sample_vecs.dot(transform_mat)

	if args.cluster == "kmeans":
		clusterer = MiniBatchKMeans(n_clusters=args.num_clusters, batch_size=int(.15*args.num_samples))
		clusterer.fit(sample_vecs)
		print clusterer.inertia_
		np.savez(args.outfile, cluster_centers=clusterer.cluster_centers_, labels=clusterer.labels_, sample_inds=sample_inds, transform_mat=transform_mat)
		import matplotlib.pyplot as plt
		plt.hist(clusterer.labels_, bins=args.num_clusters)
		plt.show()
	#Used for the Fischer vector
	elif args.cluster == "gmm":
		from sklearn.mixture import GaussianMixture
		mixture = GaussianMixture(n_components=args.num_clusters, covariance_type='diag', max_iter=200, verbose=2, n_init=5)
		mixture.fit(sample_vecs)
		print mixture.means_.shape, mixture.covariances_.shape, mixture.weights_.shape
		np.savez(args.outfile, transform_mat=transform_mat, cluster_centers=mixture.means_, variances=mixture.covariances_, weights=mixture.weights_)
	else:
		raise ValueError("Unrecognized clustering type {}".format(args.cluster))
#	print data['features'].shape, data['lengths'].shape, data['files'].shape
	

