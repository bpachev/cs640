from os import listdir
from argparse import ArgumentParser, FileType
import numpy as np
from features import grayscale, get_sift, get_hog, lbp_transform, get_segmentation
from sklearn.neighbors import BallTree
from sklearn.decomposition import PCA

"""
A file to compute features for all images and store in an npz archive.
This allows us to save a lot of computational time when experimenting with different approaches and combinations of features.

"""
def is_train(f):
	f = str(f)
	return int(f.split(".")[0][-1]) % 2

def file_class(f, binary=False):
	if binary: return int("birds" not in f)
	return int(str(f).split("/")[-1][:3])

def category_info(files,binary):
	train_mask = np.array([is_train(f) for f in files], dtype=np.bool)
	labels = np.array([file_class(f,binary) for f in files], dtype=np.int32)
	return train_mask, labels

def sift_interest(args, progress_iters=50):
	feat_list = []
	lengths = np.zeros(len(args.files), dtype=np.int64)
	kp_list = []
	from cv2 import KeyPoint_convert
	for i,f in enumerate(args.files):
		img = grayscale(f)
		mask = None
		if args.segment: mask = get_segmentation(f)
		kp, feats = get_sift(img, mask)
		lengths[i] = len(kp)
		kp_coords = KeyPoint_convert(kp)
		kp_coords[:,0] /= img.shape[1]
		kp_coords[:,1] /= img.shape[0]
		kp_list.append(kp_coords)
		feat_list.append(feats)
		if i and i%progress_iters == 0: print "Processed image ",i

	return {'features':np.vstack(feat_list), 'lengths':lengths, 'kp_coords':np.vstack(kp_list)}

def compute_histogram(image_features, tree):
	#Try brute-forcing the nearest center
	dist, inds = tree.query(image_features)
	return np.bincount(inds.flatten())

from math import floor
def compute_spatial_pyramid(image_features, tree, kp_coords, levels, num_feature_types):
	dist, inds = tree.query(image_features)
	inds = inds.flatten()
	#Level number i adds 4^i * num_feature_types features
	#In total there will be 4^(levels+1)/3 * num_feature_types features
	#First we need to compute histograms for each feature on the finest level of accuracy
	#We need to map each keypoint to the region it corresponds to
	#Basically use a base-4 representation
	N = 2**levels
	counts = np.zeros((N,N,num_feature_types), dtype=np.int32)
	#Need to convert the keypoint relative coordinates in the image to integers
	coord_indices = np.floor(N*kp_coords-1e-10).astype(np.int32) #Just in case
	for i in xrange(len(inds)):
		counts[coord_indices[i][0],coord_indices[i][1], inds[i]] += 1
	
	arrs = [counts.flatten()]
	for level in range(levels)[::-1]:
		#Sum over every group of four blocks
		counts = counts[::2,::2,:] + counts[::2,1::2,:] + counts[1::2,::2,:] + counts[1::2,1::2,:]
		arrs.append(counts.flatten())
	return np.hstack(arrs)

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
	if 'kp_coords' in feature_data and args.pyramid_levels:
		kp_coords = feature_data['kp_coords']
	final_feats = np.zeros((num_pics, (4**(1+args.pyramid_levels)-1)/3*num_bins), dtype=np.int64)

	inds = np.cumsum(lengths[:num_pics])
	start_ind = 0
	for i in xrange(len(inds)):
		if i and i%progress_iters == 0: print "Completed iteration ",i
		start_ind, inds[i]
		temp = features[start_ind:inds[i]].dot(transform_mat)
		if args.pyramid_levels:
			hist = compute_spatial_pyramid(temp, tree, kp_coords[start_ind:inds[i]], args.pyramid_levels, num_bins)
		else:
			hist = compute_histogram(temp, tree)
		
		final_feats[i,:len(hist)] = hist
		start_ind = inds[i]
	return {'features':final_feats, 'labels':feature_data['labels'], 'train_mask':feature_data['train_mask']}

def fisher(args, progress_iters=20):
	if args.input_features is None or args.cluster_info is None:
		raise ValueError("To do compute Fisher vectors, you need to specify the cluster information (must be gmm) and the input features file")
	feature_data = np.load(args.input_features)
	
	features = feature_data['features']
	lengths = feature_data['lengths']

	mixture = np.load(args.cluster_info)
	transform_mat = mixture['transform_mat'] if 'transform_mat' in mixture else None

	means = mixture['cluster_centers']
	variances = mixture['variances']
	weights = mixture['weights']
	num_pics = len(lengths)
	inds = np.cumsum(lengths[:num_pics])
	start_ind = 0
	num_comps = means.shape[0]
	d = features.shape[1]
	final_feats = np.zeros((num_pics, d * 2 * num_comps)) 
	print means[0], variances[0]
	for i in xrange(len(inds)):
		if i and i%progress_iters == 0: print "Completed iteration ",i
		start_ind, inds[i]
		temp = features[start_ind:inds[i]].dot(transform_mat)
		#Need to compute the q-matrix of weights. It ought to depend on the weights of the distribution
		#In practice, each vector is so solidly in one distribution that it makes no sense to compute the other qk
		#So we use the 'fast' implementation and pick the component with maximal log-probability
		mu_mat = np.zeros((num_comps, d))
		sig_mat = np.zeros((num_comps, d))
		for j in xrange(lengths[i]):
			x = temp[j]
			probs = np.sum(-.5*(means-x)**2 * (1./variances), axis=1) + np.log(weights)
			mode = np.argmax(probs)
			mu_mat[mode] += (x-means[mode])/variances[mode]
			sig_mat[mode] += ((x-means[mode])/variances[mode])**2 - 1

		mu_mat *= (1./lengths[i] * weights ** (-.5)).reshape((num_comps, 1))
		sig_mat *= (1./lengths[i] * (2*weights)**-.5).reshape((num_comps, 1))
		vec = np.hstack([mu_mat.flatten(), sig_mat.flatten()])
		vec = np.sign(vec)*np.abs(vec) ** .5
		vec /= np.sqrt(np.dot(vec,vec))
		final_feats[i] = vec
		start_ind = inds[i]
#	print final_feats.shape
	train_mask, labels = category_info(args.files, args.binary)
	return {'features':final_feats}
	

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
	train_mask, labels = category_info(args.files, args.binary)
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
	parser.add_argument("--pyramid-levels", type=int, default=0)
	parser.add_argument("--segment", action="store_true")
	parser.add_argument("--binary",action="store_true")
	
	args = parser.parse_args()
	
	args.files = [args.dirname+"/"+f for f in listdir(args.dirname)]
	if args.binary:
		args.files += ["birds/" + f for f in listdir("birds")]
	
	actions = {"sift_interest":sift_interest,"quantize":quantize, "hog":hog,"lbp":lbp, 'fisher':fisher}
	
	if args.feature_type not in actions: 
		raise ValueError("Unsupported or unrecognized feature type {}".format(args.feature_type))
	
	info = actions[args.feature_type](args)
	if 'train_mask' not in info or 'labels' not in info:
			train_mask, labels = category_info(args.files, args.binary)
			info['labels'] = labels
			info['train_mask'] = train_mask

	np.savez(args.outfile, **info)
