import numpy as np
from argparse import ArgumentParser, FileType
from sklearn.svm import LinearSVC, SVC
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.calibration import CalibratedClassifierCV
import xgboost as xgb
import matplotlib.pyplot as plt


def laymans_confusion_matrix(true, pred):
	true = true.astype(np.int32)
	pred = pred.astype(np.int32)
	labels = sorted(list(set([l for l in true]+[l for l in pred])))
	label_dict = {}
	for i, label in enumerate(labels):
		label_dict[label] = i
	nlabels = len(labels)
	mat = np.zeros((nlabels, nlabels))
	for t, p in zip(true, pred):
		mat[label_dict[t],label_dict[p]] += 1
	return mat/np.sum(mat,axis=1)

def precision_recall(labels, probs):
	#Need to use cumsum a lot
	#First, sort the labels by probability
	labels += np.max(labels)-np.min(labels)
	mask = np.argsort(probs)
	sorted_labels = labels[mask]
	sorted_probs = probs[mask]
	num_positives = np.cumsum(sorted_labels)
	#True positives + false negatives = the number of positively labeled data points
	recall_denom = num_positives[-1]
	#Precision denom = simply the total number of examples classified as positive
	true_positives = recall_denom - num_positives
	recall = true_positives / float(recall_denom)
	recall = recall[::-1]
	recall_vals, inds = np.unique(recall, return_index=True)
	precision = true_positives.astype(np.float64)
	precision[:-1] /= 1.*np.arange(1,len(labels))[::-1]
	precision[-1] = 1
	from sklearn.metrics import precision_recall_curve
	return recall_vals, precision[::-1][inds]

def auc(x,y):
	weights = np.zeros(len(x))
	diffs = x[1:]-x[:-1]
	weights[:-1] = diffs
	weights[1:] += diffs
	return np.dot(y, weights/2.)


def get_dataset(feature_files):
	train_mask, labels = None, None
	features = []
	for f in feature_files:
		ark = np.load(f)
		if labels is not None and not np.allclose(labels,ark['labels']): raise ValueError("You are using two datasets with inconsistent labels")
		elif labels is None: labels = ark['labels']

		if train_mask is not None and not np.allclose(train_mask,ark['train_mask']): raise ValueError("You are using two datasets with inconsistent train-test split")
		elif train_mask is None: train_mask = ark['train_mask']
	
		features.append(ark['features'])
	
	final_mat = np.hstack(features)
	test_mask = ~train_mask
	return final_mat[train_mask], labels[train_mask], final_mat[test_mask], labels[test_mask]

def plot_precision_recall(labels, probs):
	recall, precision = precision_recall(labels, probs)
	plt.plot(recall, precision)
	plt.title("AUC = {}".format(auc(recall,precision)))
	plt.xlabel("Recall")
	plt.ylabel("Precision")
	plt.show()


def svc(args, linear=True):
	trainx, trainy, testx, testy = get_dataset(args.inputfiles)
	if linear:
		model = LinearSVC(C=.001)
	else:
		model = SVC(C=100)
	
#	reducer = PCA(n_components = 400)
#	trainx = reducer.fit_transform(trainx)
#	testx = reducer.transform(testx)

	#This results in a significant improvement
	scaler = StandardScaler()
	trainx = scaler.fit_transform(trainx)
	testx = scaler.transform(testx)
	model.fit(trainx, trainy)
	print model.score(testx, testy), "APR"
	preds = model.predict(testx)
	print laymans_confusion_matrix(testy, preds)
	if max(testy) == 1:
		plot_precision_recall(testy, model.decision_function(testx))

def test_xgb(args):
	trainx, trainy, testx, testy = get_dataset(args.inputfiles)
	trainy -= 1
	testy -= 1
	dtrain = xgb.DMatrix(trainx, trainy)
	dtest = xgb.DMatrix(testx, testy)
	
	params = {"objective":"multi:softmax", "num_class":10, 'eta':.5, 'colsample_bytree':.9}
	bst = xgb.train(params, dtrain, num_boost_round=500)
#	model = LinearSVC(C=.005)
#	scaler = StandardScaler()
#	trainx = scaler.fit_transform(trainx)
#	testx = scaler.transform(testx)
	pred = bst.predict(dtest)
#	print model.score(testx, testy), "APR"
	print np.mean(pred==testy)
	print laymans_confusion_matrix(testy, pred)

def test_sklearn_xgb(args):
	trainx, trainy, testx, testy = get_dataset(args.inputfiles)
	trainy -= 1
	testy -= 1
	from sklearn.multiclass import OneVsRestClassifier
	model = OneVsRestClassifier(xgb.XGBClassifier(njobs=-1, eta=.3, num_boost_round=200))
	model.fit(trainx, trainy)
	pred = model.predict(testx)
	print np.mean(pred==testy)
	print laymans_confusion_matrix(testy, pred)
	if max(testy)-min(testy) == 1:
		probs = model.predict_proba(testx)[:,1].flatten()
		print probs.shape, testy.shape
		plot_precision_recall(testy, probs)


def voting_classifier(args):
	trainx, trainy, testx, testy = get_dataset(args.inputfiles)
	from sklearn.ensemble import VotingClassifier
	scaler = StandardScaler()
	trainx = scaler.fit_transform(trainx)
	testx = scaler.transform(testx)
	model = VotingClassifier([('xgb',xgb.XGBClassifier(njobs=-1, eta=.5, num_boost_round=200)), ('linearsvc',CalibratedClassifierCV(LinearSVC(C=.001))), ('svc',SVC(C=10, probability=True))], voting='soft')
	model.fit(trainx, trainy)
	pred = model.predict(testx)
	print np.mean(pred==testy)
	print laymans_confusion_matrix(testy, pred)
	if max(testy)-min(testy) == 1:
		probs = model.predict_proba(testx)[:,1].flatten()
		print probs.shape, testy.shape
		plot_precision_recall(testy, probs)


if __name__ == "__main__":
	parser = ArgumentParser()
	parser.add_argument("inputfiles", nargs="+", type=FileType('r'))
	#For now, do a simple train-test split

	args = parser.parse_args()
	voting_classifier(args)
#	svc(args, linear=True)
#	test_sklearn_xgb(args)
#	print trainx, trainy
#	print testx, testy
