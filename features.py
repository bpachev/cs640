import cv2 as cv
import numpy as np

def grayscale(fname):
	img = cv.imread(fname)
	res = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
	return res

def lbp_transform(img):
	#We compute a simple lbp transform using 8 surrounding pixels
	#img is assumed to be grayscale
	#Our implementation for sake of efficiency results in a nonstandard bit-mapping
	# 5 6 7
	# 4 v 0
	# 3 2 1
	# Bits 4,5,6 and 7 are set if v >= to its neighbor, while for 0-3 we must have v > neighbor
	# This allows us to reduce the number of comparisons by 2.
	# For example, if v is not greater than its neighbor to the right, then its neighbor to the right is greater than or equal to it.
	# So we can set bit 4 for v's neighbor. One comparison gave two pieces of information.
	h,w = img.shape
	offsets = [(0,1),(1,1),(1,0),(1,-1)]
	bit_masks = np.zeros(img.shape, dtype=np.int64)
	for i, offset in enumerate(offsets):
		#Create a bitmask
		h_offset, w_offset = offset
		if h_offset < 0:
			h_start = -h_offset
			h_stop = h
		else:
			h_start = 0
			h_stop = h - h_offset

		if w_offset < 0:
			w_start = -w_offset
			w_stop = w
		else:
			w_start = 0
			w_stop = w - w_offset
		
		mask = img[h_start:h_stop, w_start:w_stop] > img[h_start+h_offset:h_stop+h_offset, w_start+w_offset:w_stop+w_offset]
		bit_masks[h_start:h_stop,w_start:w_stop] += 2**i * mask
		bit_masks[h_start+h_offset:h_stop+h_offset, w_start+w_offset:w_stop+w_offset] += 2**(i+4) * (1-mask)
	
	print bin(bit_masks[-2,-2])
	print img[-3:,-3:]
	return np.bincount(bit_masks.flatten())

sift = cv.xfeatures2d.SIFT_create()
def get_sift(img, mask=None):
	return sift.detectAndCompute(img, mask)

#from sys import argv
#gray = grayscale(argv[1])
#print lbp_transform(gray)
#kp, feats = get_sift(gray)
#print len(kp), len(feats)
