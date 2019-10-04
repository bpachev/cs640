import cv2 as cv
import numpy as np
import skimage.feature as sk


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
	
	return np.bincount(bit_masks.flatten())

sift = cv.xfeatures2d.SIFT_create()
def get_sift(img, mask=None):
	return sift.detectAndCompute(img, mask)

#winSize = (768,1024)
#blockSize = (128,128)
#blockStride = (64,64)
#cellSize = (32,32)
#nbins = 9
#print hog.winSize
#print hog.cellSize
#print hog.blockSize, hog.blockStride
def get_hog(img, shape = None):
	if shape is not None:
		img = cv.resize(img, shape)
	return sk.hog(img, pixels_per_cell=(32,32), cells_per_block=(2,2),block_norm='L2-Hys').flatten()

def get_segmentation(fname):
	parts = fname.split("/")
	img_num = parts[-1].split(".")[0]
	real_fname = img_num+"_mask.png"
	mask = grayscale("/".join(parts[:-2]+["segmentations",real_fname]))
	mask[mask>0] = 1
	return mask.astype(np.uint8)

if __name__ == "__main__":
	from sys import argv
	img = grayscale(argv[1])
	kp, feats = get_sift(img)
	print feats.shape
	kp, feats = get_sift(img, get_segmentation(argv[1]))
	print feats.shape
#print get_hog(gray)
#print lbp_transform(gray)
#kp, feats = get_sift(gray)
#print len(kp), len(feats)

