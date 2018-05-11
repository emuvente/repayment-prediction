# Load libraries
import pandas
import numpy as np
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier

# Load dataset
dataset = pandas.read_csv('file://localhost/Volumes/kiva/data-analysis/fund_time_prediction/expiration_prediction/loan_data.csv')

# Split-out validation dataset
array = dataset.values
X = array[:,0:17]
# Y = np.around(array[:,17] * 0.015)
Y = array[:,17] == 100
validation_size = 0.20
seed = 7
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)

# Test options and evaluation metric
seed = 7
scoring = 'accuracy'

# # Spot Check Algorithm
# kfold = model_selection.KFold(n_splits=10, random_state=seed)
# cv_results = model_selection.cross_val_score(DecisionTreeClassifier(), X_train, Y_train, cv=kfold, scoring=scoring)
# msg = "%s: %f (%f)" % ('CART', cv_results.mean(), cv_results.std())
# print(msg)

# Make predictions on validation dataset
cart = DecisionTreeClassifier()
cart.fit(X_train, Y_train)
predictions = cart.predict(X_validation)
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))

estimator = cart

# The decision estimator has an attribute called tree_  which stores the entire
# tree structure and allows access to low level attributes. The binary tree
# tree_ is represented as a number of parallel arrays. The i-th element of each
# array holds information about the node `i`. Node 0 is the tree's root. NOTE:
# Some of the arrays only apply to either leaves or split nodes, resp. In this
# case the values of nodes of the other type are arbitrary!
#
# Among those arrays, we have:
#   - left_child, id of the left child of the node
#   - right_child, id of the right child of the node
#   - feature, feature used for splitting the node
#   - threshold, threshold value at the node
#

# Using those arrays, we can parse the tree structure:

n_nodes = estimator.tree_.node_count
children_left = estimator.tree_.children_left
children_right = estimator.tree_.children_right
feature = estimator.tree_.feature
threshold = estimator.tree_.threshold
node_value = estimator.tree_.value
tree_classes = estimator.classes_

classified_values = []
for i in range(n_nodes):
	totval = np.sum(node_value[i])
	str = ""
	for j in range(len(tree_classes)):
		val = round(10000 * (node_value[i][0][j] / totval))/100
		str = "%s%s: %s%s    " % (str, tree_classes[j], val, '%')
	classified_values.append(str)

# The tree structure can be traversed to compute various properties such
# as the depth of each node and whether or not it is a leaf.
node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
is_leaves = np.zeros(shape=n_nodes, dtype=bool)
stack = [(0, -1)]  # seed is the root node id and its parent depth
while len(stack) > 0:
	node_id, parent_depth = stack.pop()
	node_depth[node_id] = parent_depth + 1

	# If we have a test node
	if (children_left[node_id] != children_right[node_id]):
		stack.append((children_left[node_id], parent_depth + 1))
		stack.append((children_right[node_id], parent_depth + 1))
	else:
		is_leaves[node_id] = True

print("The binary tree has %s nodes and has "
	  "the following structure:"
	  % n_nodes)
for i in range(min(n_nodes, 50)):
	if is_leaves[i]:
		print("%s%s: %s"
			  % (node_depth[i] * " ",
				 i,
				 tree_classes[np.argmax(node_value[i])]
				 # node_value[i]
				 # classified_values[i]
				 ))
	else:
		print("%s%s: if %s <= %s go to %s else %s"
			  % (node_depth[i] * " ",
				 i,
				 dataset.axes[1][feature[i]],
				 threshold[i],
				 children_left[i],
				 children_right[i],
				 ))
print()

# First let's retrieve the decision path of each sample. The decision_path
# method allows to retrieve the node indicator functions. A non zero element of
# indicator matrix at the position (i, j) indicates that the sample i goes
# through the node j.

node_indicator = estimator.decision_path(X_validation)

# Similarly, we can also have the leaves ids reached by each sample.

leave_id = estimator.apply(X_validation)

# Now, it's possible to get the tests that were used to predict a sample or
# a group of samples. First, let's make it for the sample.

sample_id = 0
node_index = node_indicator.indices[node_indicator.indptr[sample_id]:
									node_indicator.indptr[sample_id + 1]]

print('Rules used to predict sample %s: ' % sample_id)
for node_id in node_index:
	if leave_id[sample_id] != node_id:
		continue

	if (X_validation[sample_id, feature[node_id]] <= threshold[node_id]):
		threshold_sign = "<="
	else:
		threshold_sign = ">"

	print("decision id node %s : (X_test[%s, %s] (= %s) %s %s)"
		% (node_id,
			sample_id,
			dataset.axes[1][feature[node_id]],
			X_validation[sample_id, feature[node_id]],
			threshold_sign,
			threshold[node_id]))

# For a group of samples, we have the following common node.
sample_ids = [0, 1, 2, 3]
common_nodes = (node_indicator.toarray()[sample_ids].sum(axis=0) ==
				len(sample_ids))

common_node_id = np.arange(n_nodes)[common_nodes]

print("\nThe following samples %s share the node %s in the tree"
	% (sample_ids, common_node_id))
print("It is %s %% of all nodes." % (100 * len(common_node_id) / n_nodes,))
