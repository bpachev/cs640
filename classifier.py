import numpy as np
from argparse import ArgumentParser, FileType
from sklearn.svm import LinearSVC, SVC
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

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

def svc(args):
	trainx, trainy, testx, testy = get_dataset(args.inputfiles)
	model = LinearSVC(C=.005)
#	scaler = StandardScaler()
#	trainx = scaler.fit_transform(trainx)
#	testx = scaler.transform(testx)
	model.fit(trainx, trainy)
	print model.score(testx, testy), "APR"
	print confusion_matrix(testy, model.predict(testx))
	

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
	print confusion_matrix(testy, pred)

def test_sklearn_xgb(args):
	trainx, trainy, testx, testy = get_dataset(args.inputfiles)
	trainy -= 1
	testy -= 1
	from sklearn.multiclass import OneVsRestClassifier
	model = OneVsRestClassifier(xgb.XGBClassifier(njobs=-1, eta=.5, num_boost_round=200))
	model.fit(trainx, trainy)
	pred = model.predict(testx)
	print np.mean(pred==testy)
	print confusion_matrix(testy, pred)


if __name__ == "__main__":
	parser = ArgumentParser()
	parser.add_argument("inputfiles", nargs="+", type=FileType('r'))
	#For now, do a simple train-test split

	args = parser.parse_args()
	test_sklearn_xgb(args)
#	print trainx, trainy
#	print testx, testy
