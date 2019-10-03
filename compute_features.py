from os import listdir
from argparse import ArgumentParser, FileType
import numpy as np
from features import grayscale, get_sift, get_hog, lbp_transform
from sklearn.neighbors import BallTree
from sklearn.decomposition import PCA

"""
A file to compute features for all images and store in an npz archive.
This allows us to save a lot of computational time when experimenting with different approaches and combinations of features.

"""
def is_train(f):
	f = str(f)
	return int(f.split(".")[0][-1]) % 2

def file_class(f):
	return int(str(f).split("/")[-1][:3])

def category_info(files):
	train_mask = np.array([is_train(f) for f in files], dtype=np.bool)
	labels = np.array([file_class(f) for f in files], dtype=np.int32)
	return train_mask, labels

def sift_interest(args, progress_iters=50):
	feat_list = []
	lengths = np.zeros(len(args.files), dtype=np.int64)

	for i,f in enumerate(args.files):
		img = grayscale(f)
		kp, feats = get_sift(img)
		lengths[i] = len(kp)
		feat_list.append(feats)
		if i and i%progress_iters == 0: print "Processed image ",i

	return {'features':np.vstack(feat_list), 'lengths':lengths}

def compute_histogram(image_features, tree):
	#Try brute-forcing the nearest center
	dist, inds = tree.query(image_features)
	return np.bincount(inds.flatten())

def quantize(args, progress_iters=20):
	if args.input_features is None or args.cluster_info is None:
		raise ValueError("To do vector quantization, you need to specify the cluster information and the features file")
	
	cluster_data = np.load(args.cluster_info)
	transform_mat = cluster_data['transform_mat']
	feature_data = np.load(args.input_features)
	
	#Need to 
	features = feature_data['features']
	lengths = feature_data['lengths']
	num_pics = len(lengths)
	centers = cluster_data['cluster_centers']
	tree = BallTree(centers)
	num_bins = centers.shape[0]
	final_feats = np.zeros((num_pics, num_bins), dtype=np.int64)

	inds = np.cumsum(lengths[:num_pics])
	start_ind = 0
	for i in xrange(len(inds)):
		if i and i%progress_iters == 0: print "Completed iteration ",i
		start_ind, inds[i]
		hist = compute_histogram(features[start_ind:inds[i]].dot(transform_mat), tree)
		final_feats[i,:len(hist)] = hist
		start_ind = inds[i]
	return {'features':final_feats, 'labels':feature_data['labels'], 'train_mask':feature_data['train_mask']}

def hog(args):
	mat = None
	nfiles = len(args.files)
	for i,f in enumerate(args.files):
		vec = get_hog(grayscale(f), shape=(512,512))
		if mat is None:
			mat = np.zeros((nfiles, vec.size))
		mat[i,:] = vec
		if i and i % 50 == 0: print "Completed ",i
	#There are usually a ridiculous amount of HOG features
	#Lets chop the dimensionality down a peg or two
	train_mask, labels = category_info(args.files)
	reducer = PCA(n_components = 400)
	reducer.fit(mat[train_mask])
	feats = reducer.transform(mat)
	return {'features':feats}

def lbp(args):
	nfiles = len(args.files)
	feats = np.zeros((nfiles, 256), dtype=np.int32)
	for i,f in enumerate(args.files):
		vec = lbp_transform(grayscale(f))
		feats[i,:vec.size] = vec
		if i and i % 50 == 0: print "Completed ",i

	print feats.shape
	return {'features':feats}

if __name__ == "__main__":
	parser = ArgumentParser()
	parser.add_argument("--dirname", default="leedsbutterfly/images")
	parser.add_argument("outfile",type=FileType('w'))
	parser.add_argument("--feature-type", default="sift_interest")
	#If we are computing features which are derived from simpler features (e.g. vector quantization), then we need an input file
	parser.add_argument("--input-features", type=FileType('r'))
	#if doing quantization, we need the precomputed cluster centers (and other information)
	parser.add_argument("--cluster-info", type=FileType('r'))
	
	args = parser.parse_args()
	
	args.files = [args.dirname+"/"+f for f in listdir(args.dirname)]
	
	actions = {"sift_interest":sift_interest,"quantize":quantize, "hog":hog,"lbp":lbp}
	
	if args.feature_type not in actions: 
		raise ValueError("Unsupported or unrecognized feature type {}".format(args.feature_type))
	
	info = actions[args.feature_type](args)
	if 'train_mask' not in info or 'labels' not in info:
			train_mask, labels = category_info(args.files)
			info['labels'] = labels
			info['train_mask'] = train_mask

	np.savez(args.outfile, **info)
