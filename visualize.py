import numpy as np
import argparse as ap 
import matplotlib.pyplot as plt

def ordered_balanced_sample(y, samples_per_class=20):
	"""
	y -- a numpy array with k unique values
	samples_per_class -- an integer specifying how many samples to draw for each unique value
	Returns:
		inds -- indices to get the samples in order

	"""

	sorted_inds = np.argsort(y)
	vals, inds = np.unique(y[sorted_inds], return_index=True)
	num_classes = len(vals)

	if samples_per_class * num_classes > len(y): raise ValueError("Too many samples required {}*{} > {} !".format(samples_per_class, num_classes,len(y)))
	
	res = np.zeros(samples_per_class * num_classes, dtype=np.int64)
	for i in xrange(0,num_classes*samples_per_class, samples_per_class):
		j = inds[int(i/samples_per_class)]
		res[i:i+samples_per_class] = sorted_inds[j:j+samples_per_class]
	return res

	

def visualize_histograms(ark):
	samples = 20
	mask = ordered_balanced_sample(ark['labels'],samples_per_class=samples)
	mat = ark['features'][mask].T
	plt.subplot(121)
	plt.imshow(mat)
	plt.subplot(122)
	
	total_classes = int(mat.shape[1]/samples)
	for i in xrange(total_classes):
		mat[:,i*samples:(i+1)*samples] = np.mean(mat[:,i*samples:(i+1)*samples], axis=1).reshape((400,1))
	plt.imshow(mat)
	plt.show()

if __name__ == "__main__":
	parser = ap.ArgumentParser()
	parser.add_argument("infile", type=ap.FileType('r'))
	parser.add_argument("--mode", type=str, nargs="?", default="histograms")

	args = parser.parse_args()
	ark = np.load(args.infile)
	if args.mode == "histograms":
		visualize_histograms(ark)
	elif args.mode == "words":
		pass
	else:
		raise ValueError("Unrecognized visualization {}".format(args.mode))
	
