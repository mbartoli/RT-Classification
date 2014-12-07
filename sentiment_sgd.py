
# coding: utf-8

# In[1]:

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion
from sklearn.grid_search import GridSearchCV
from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import chi2
import numpy as np
import csv
import sys


# In[2]:

X_train, y_train, ID_test, X_test = [], [], [], []


# In[3]:

with open('train.tsv', 'r') as f:
	f.readline()
	csvreader = csv.reader(f, delimiter='\t')
	for row in csvreader:
		X_train.append(row[2])
		y_train.append(row[3])


# In[4]:

with open('test.tsv', 'r') as f:
	f.readline()
	csvreader = csv.reader(f, delimiter='\t')
	for row in csvreader:
		ID_test.append(row[0])
		X_test.append(row[2])


# In[6]:

words = TfidfVectorizer(analyzer="word", binary=False, use_idf=True, stop_words="english", min_df=3)
char = TfidfVectorizer(analyzer="char", binary=False, use_idf=True)


# In[7]:

select = SelectPercentile(score_func=chi2)


# In[9]:

feat = FeatureUnion([('words', words),('char', char)])


# In[12]:

text_clf = Pipeline([('feat', feat),('clf', SGDClassifier(penalty='l2'))])


# In[13]:

parameters = {'feat__words__ngram_range': [(1,5), (1,6)],
              'feat__words__min_df': (2,3),
              'feat__words__use_idf': (True, False),
              'feat__char__use_idf': (True, False),
              'clf__alpha': (.00001, .000001),
              'clf__loss': ("hinge", "log", "modified_huber")
}


# In[14]:

gs_clf = GridSearchCV(text_clf, parameters, cv=2, verbose=True, n_jobs=-2)


# In[15]:

gs_clf.fit(X_train, y_train)


# In[16]:

predictions = gs_clf.predict(X_test)


# In[18]:

with open('scores.csv', 'w') as outfile:
	csvwriter = csv.writer(outfile, delimiter=',')
	header = ["mean","std"]
	param_names = [param for param in gs_clf.param_grid]
	header.extend(param_names)
	csvwriter.writerow(header)
	for config in gs_clf.grid_scores_:
		mean = config[1]
		std = np.std(config[2])
		row = [mean,std]
		params = [str(p) for p in config[0].values()]
		row.extend(params)
		csvwriter.writerow(row)


# In[20]:

best_parameters, score, _ = max(gs_clf.grid_scores_, key=lambda x: x[1])


# In[21]:

for param_name in sorted(parameters.keys()):
    print("%s: %r" % (param_name, best_parameters[param_name]))
print score


# In[22]:

with open('submission_sgd.csv', 'w') as outfile:
	outfile.write("PhraseId,Sentiment\n")
	for phrase_id,pred in zip(ID_test,predictions):
		outfile.write('{},{}\n'.format(phrase_id,pred))


# In[ ]:



