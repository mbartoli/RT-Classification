"""

@author Michael Bartoli

classify.py: trains a linear support vector machine to classify movie reviews

Args:
	sys[1]:	path to top-level of project directory

"""

import sys
import numpy as np
from sklearn.datasets import load_files
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn import metrics
from sklearn.pipeline import Pipeline

def main(path_to_data):
	categories = ["0","1","2","3","4"]
	dataset = load_files(path_to_data, shuffle=False)
	print("n_samples: %d" % len(dataset.data))
	docs_train, docs_test, y_train, y_test = train_test_split(dataset.data, dataset.target, test_size=0.25, random_state=None)
	pipeline = Pipeline([('vect', CountVectorizer()),('tfidf', TfidfTransformer()),('clf', LinearSVC(C=1000)),])
	_ = pipeline.fit(dataset.data, dataset.target)
	predicted = pipeline.predict(docs_test)
	print np.mean(predicted == dataset.target)
	
	"""
	parameters = {
		'vect__ngram_range': [(1, 1), (1, 2)],
	}
	grid_search = GridSearchCV(pipeline, parameters, n_jobs=-1)
	grid_search.fit(docs_train, y_train)
	print(grid_search.grid_scores_)
	y_predicted = grid_search.predict(docs_test)
	print(metrics.classification_report(y_test, y_predicted,target_names=dataset.target_names))
	cm = metrics.confusion_matrix(y_test, y_predicted)
	print(cm)
	"""	

if __name__ == "__main__":
	path = sys.argv[1]
	main(path)
