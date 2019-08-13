import numpy as np
from random import randrange, shuffle
from collections import Counter
from scipy import spatial, stats
from itertools import islice

class random_forest():

    def __init__(self, train_data, n_folds, max_depth, min_size, sample_size, n_trees, n_features, test_ratio):
        '''
        :param train_data: numpy array
        :param n_folds: int, Split a dataset into n folds, the first n-1 folds are training and the n-th fold is testing
        cross-validation process repeat n times
        :param max_depth: int, the maximum depth of the decision tree
        :param min_size: int, the minimum number of element in a branch
        :param sample_size: float, the percentage of data being selected for training
        :param n_trees: int, the number of trees per cross-validation
        :param n_features: int, the number of features
        :param test_ratio: float, the ratio of training data in original data
        The code is modified from the code by Siraj Raval
        https://github.com/llSourcell/random_forests/blob/master/Random%20Forests%20.ipynb

        '''
        self.test_ratio = test_ratio
        self.train_data = train_data.ix[:int(len(train_data) * test_ratio), 1:].values
        self.temp_test = train_data.ix[int(len(train_data) * test_ratio):, 1:].values
        self.n_folds = n_folds
        self.max_depth = max_depth
        self.min_size = min_size
        self.sample_size = sample_size
        self.n_trees = n_trees
        self.n_features = n_features
        self.scores, self.predicted, self.forest = self.evaluation_algo(self.train_data, self.random_forest, n_folds,
                                                                        max_depth, min_size, sample_size,
                                                                        n_trees, n_features)
        print("n_features:", self.n_features)
        print('Scores: ', self.scores)
        print('Mean Accuracy: ', (sum(self.scores) / float(len(self.scores))))

    def best_trained_tree(self):
        '''
        Cross Validation is implemented here, and the best trees from multiple training trees are selected
        '''
        max_score = 0
        self.index = -1
        for i in range(len(self.forest)):
            score = 0
            predictions = [self.bagging_predict(self.forest[i], row) for row in self.temp_test]
            actual = [row[-1] for row in self.temp_test]
            for j in range(len(predictions)):
                if predictions[j] == actual[j]:
                    score += 1
            score = score / float(len(self.temp_test))
            if score > max_score:
                max_score = score
                self.index = i
            print(i, ") accuracy:", score)
        print("The most accurate tree:", self.index)

    def predict_test_data(self, test_data):
        '''
        This function serves as a wrapper to the random forest classification functionality
        :param test_data: numpy.ndarray, the imputed test data
        :return: list of predictions
        '''
        max_trees = self.forest[self.index]
        predictions = [self.bagging_predict(max_trees, row) for row in test_data]
        return predictions



    def split_dataset(self, dataset, num_folds):
        '''
        Partition data from dataset into num_folds groups
        :param dataset: numpy.ndarray
        :param num_folds: integer
        :return: list of split data
        '''
        res = [[] for n in range(num_folds)]
        ls = [n for n in range(num_folds)]
        for i in range(len(dataset)):
            temp = i % num_folds
            if temp == 0:
                shuffle(ls)
            res[ls[temp]].append(dataset[i])
        return res

    def accuracy_comp(self, actual, predict):
        '''
        Compare actual data and predict data
        :param actual: list of data
        :param predict: list of data
        :return: float, compared accuracy
        '''
        correct = 0
        for i in range(len(actual)):
            if actual[i] == predict[i]:
                correct += 1
        return correct / float(len(actual))

    def evaluation_algo(self, dataset, algorithm, n_folds, *args):
        '''
        Train data, and use trained data to estimate accuracy of the model
        :param dataset: imputed data
        :param algorithm: string, as only random forest is implemented, not used here
        :param n_folds: int, n groups
        :param args: extra possible arguments
        :return: a tuple of three lists containing evaluation results
        '''
        folds = self.split_dataset(dataset, n_folds)
        scores = []
        predicted = []
        forest = []
        for i in range(n_folds):
            train_set = folds[:i] + folds[i + 1:]
            train_set = [row for fold in train_set for row in fold]
            test_set = folds[i]
            # test: 4000 127; train: 16000 127
            predict, trees = algorithm(train_set, test_set, *args)
            forest.append(trees)
            predicted.append(predict)
            actual = [row[-1] for row in folds[i]]
            accuracy = self.accuracy_comp(actual, predict)
            scores.append(accuracy)
        return scores, predicted, forest

    def test_split(self, index, value, dataset):
        '''
        According to index number, split dataset
        :param index: int, the split point
        :param value: int, threshold value
        :param dataset: numpy.ndarray
        :return: a tuple of two lists
        '''
        left = []
        right = []
        for row in dataset:
            if row[index] < value:
                left.append(row)
            else:
                right.append(row)
        return left, right


    def gini_help(self, ls, class_values):
        '''
        Calculate gini index of smallest unit
        :param ls: numpy array
        :param class_values: int, actual class value to be predicted
        :return: float, gini index of sub unit
        '''
        temp = [row[-1] for row in ls]
        num = Counter(temp)
        res = [num[n] / float(len(temp)) for n in num]
        return 1 - np.sum(np.square(res))

    def gini_index(self, groups, class_values):
        '''
        Calculate gini index
        :param groups: list of groups
        :param class_values: int, actual class value to be predicted
        :return: float, calculated gini index
        '''
        gini = 0.0
        # print("gini_index")
        size = sum([len(group) for group in groups])
        for group in groups:
            temp_gini = self.gini_help(group, class_values)
            gini += temp_gini * len(group) / float(size)
        return gini

    def get_split(self, dataset, n_features):
        '''
        Use Gini index to select the n best features.
        :param dataset: numpy.ndarray
        :param n_features: int, number of features, can be manually set
        :return: a dictionary of lists
        '''
        class_values = list(set(row[-1] for row in dataset))  # number of classes
        b_index, b_value, b_score, b_group = 999, 999, 999, None
        features = []
        while (len(features)) < n_features:
            ind = randrange(len(dataset[0]) - 1)
            if ind not in features:
                features.append(ind)
        for index in features:
            dic = {}
            for row in dataset:
                if dic.get(row[index]) == None:
                    dic[row[index]] = 1
                    groups = self.test_split(index, row[index], dataset)  # return left, right
                    gginin = self.gini_help(dataset, class_values)
                    gini = self.gini_index(groups, class_values)
                    if gini < b_score:
                        b_index, b_value, b_score, b_group = index, row[index], gini, groups
        return {'index': b_index, 'value': b_value, 'groups': b_group}

    def to_terminal(self, group):
        '''
        Determine terminal of trees
        :param group: list, data on leaves of the tree
        :return: the max value of the list
        '''
        outcomes = [row[-1] for row in group]
        return max(set(outcomes), key=outcomes.count)

    def split(self, node, max_depth, min_size, n_features, depth):
        '''
        Calculate split in building decision trees
        :param node: dictionary, representing a node in the tree
        :param max_depth: int, maximum depth of the tree
        :param min_size: int, minimum size of leaf
        :param n_features: int, n best features used in construct trees
        :param depth: int, current depth
        :return: None
        '''
        left, right = node['groups']  # checkout left and right from root
        del node['groups']
        if (len(left) == 0) or (len(right) == 0):
            node['left'] = node['right'] = self.to_terminal(left + right)
            return
        if depth >= max_depth:  # check max depth
            if len(left) == 0 or len(right) == 0:
                node['left'] = node['right'] = self.to_terminal(left + right)
            else:
                node['left'], node['right'] = self.to_terminal(left), self.to_terminal(right)
            return
        # create a terminal node if the group of rows is too small,
        if len(left) <= min_size:
            node['left'] = self.to_terminal(left)
        else:
            node['left'] = self.get_split(left, n_features)
            self.split(node['left'], max_depth, min_size, n_features, depth + 1)

        if len(right) <= min_size:
            node['right'] = self.to_terminal(right)
        else:
            node['right'] = self.get_split(right, n_features)
            self.split(node['right'], max_depth, min_size, n_features, depth + 1)

    def build_tree(self, train, max_depth, min_size, n_features):
        '''
        Build trees
        :param train: training data
        :param max_depth: int, maximum depth
        :param min_size: int, if number of data on a leaf is less than this value the branch will be terminated
        :param n_features: int, n best features used to construct tree
        :return: dictionary, root of the ree
        '''
        root = self.get_split(train, n_features)
        self.split(root, max_depth, min_size, n_features, 1)
        return root

    def predict(self, node, row):
        '''
        Based on current built tree, predict current row's class
        :param node: dictionary, node in the tree
        :param row: list, the row to be predicted
        :return: prediction which branch the row should take
        '''
        if row[node['index']] < node['value']:
            if isinstance(node['left'], dict):
                return self.predict(node['left'], row)
            else:
                return node['left']
        else:
            if isinstance(node['right'], dict):
                return self.predict(node['right'], row)
            else:
                return node['right']

    def subsample(self, dataset, ratio):
        '''
        Fetch part of the training data to build tree
        :param dataset: numpy.ndarray
        :param ratio: float, speciy which part of the data to be used
        :return: list, the samples
        '''
        sample = []
        n_sample = len(dataset) * ratio
        while len(sample) < n_sample:
            temp_ind = randrange(len(dataset))
            sample.append(dataset[temp_ind])
        return sample

    def bagging_predict(self, trees, row):
        '''
        Return the most voted answer from multiple trees
        :param trees: the trees generated
        :param row: numpy.array, current row to be predicted
        :return: maximum of predictions
        '''
        predictions = [self.predict(tree, row) for tree in trees]
        return max(set(predictions), key=predictions.count)

    def random_forest(self, train, test, max_depth, min_size, sample_size, n_trees, n_features):
        '''
        The main function body of the Random Forest class
        :param train: training dataset
        :param test: testing dataset
        :param max_depth: int, maximum depth
        :param min_size: int, minimum size of tree leaves
        :param sample_size: int, the size of the sample to be fetched
        :param n_trees: int, number of trees
        :param n_features: int, number of best features
        :return: a tuple of two lists, including predictions and generated trees
        '''
        trees = []
        for i in range(n_trees):
            sample = self.subsample(train, sample_size)
            tree = self.build_tree(sample, max_depth, min_size, n_features)
            trees.append(tree)
            print("tree (", i)
        predictions = [self.bagging_predict(trees, row) for row in test]
        return predictions, trees



def un_condense(i, j, n):
    '''
    This function is used to unpack the condense array returned by scipy.spatial.distance
    :param i: int, row index
    :param j: int, col index
    :param n: size of condense array
    :return: int, condense array index
    '''
    if i < j:
        i, j = j, i
    return int(n*j - j*(j+1)/2 + i - 1 - j)




class DataImputation():
    def __init__(self, pandas_dataframe, categorical=None, continuous=None, discrete=None):
        '''
        This class implements data imputation methods
        :param pandas_dataframe: pandas.DataFrame object, the raw data with missing values
        :param categorical: list of strings, contains all categorical attribute labels
        :param continuous: list of strings, contains all continuous attribute labels
        :param discrete: list of strings, contains all discrete attribute labels
        '''
        self.raw = pandas_dataframe
        self.categorical_attribute_names = categorical
        self.continuous_attribute_names = continuous
        self.discrete_attribute_names = discrete
        self.temp = None

    def central_tendency(self):
        '''
        Populates missing value with mean if attribute is continuous, median if attribute is discrete and mode if attribute is categorical
        :return: dict, coordinates of missing values (first level is column and second level is row)
        '''
        nan_dict = {}
        for column_name in self.raw:
            nan_dict[column_name] = self.raw[column_name].index[self.raw[column_name].apply(np.isnan)]

            if column_name in self.continuous_attribute_names:
                mean = self.raw[column_name].dropna().mean()
                self.raw[column_name].fillna(mean, inplace=True)
            elif column_name in self.discrete_attribute_names:
                median = self.raw[column_name].dropna().median()
                self.raw[column_name].fillna(median, inplace=True)
            else:
                mode = self.raw[column_name].dropna().mode()
                self.raw[column_name].fillna(mode[0], inplace=True)
        return nan_dict

    def retrieve_imputed_dataframe(self):
        return self.raw

    def normalization(self, method='z_score'):
        '''
        Normalizes COMPLETE dataframe
        :param method: specify which normalization method to use. Right now only supports z-score normalization
        :return: None
        '''
        if method == 'z_score':
            self.raw = (self.raw - np.mean(self.raw, axis=0))/np.std(self.raw, axis=0)

    def knn(self, k=None, metric='euclidean', iteration=10):
        '''
        This function implements iterative knn:
        https://www.sciencedirect.com/science/article/pii/S0164121212001586
        :param k: int, number of nearest neighbors taken into consideration. If not specified, use sqrt(number of attributes)
        :param metric: string, used to calculate distance between rows. Supported all types that scipy.spatial.distance supports
        :param iteration: int, number of iterations. Higher the more accuracy
        :return: None
        '''
        if k == None:
            k = int(np.round(np.sqrt(len(self.raw.index)), decimals=0))

        missing_dict = self.central_tendency()

        # Reverse the dictionary so that first level is row and second level is column
        reversed_missing_dict = {}
        for column, row_array in missing_dict.items():
            for row_element in row_array:
                if row_element not in reversed_missing_dict:
                    reversed_missing_dict[row_element] = [column]
                else:
                    if column not in reversed_missing_dict[row_element]:
                        reversed_missing_dict[row_element].append(column)

        # Normalize the populated dataframe using z-score normalization
        self.normalization()

        # iteration
        for not_used in range(iteration):
            print(str(not_used)+'\'s iteration')

            stored_row = self.raw.copy()
            all_coefficients = spatial.distance.pdist(X=self.raw.values, metric=metric)
            total_row = len(self.raw.index)

            for i in range(total_row):
                if i in reversed_missing_dict:
                    temp_dict = {j: all_coefficients[un_condense(i, j, total_row - 1)] for j in range(total_row) if j != i}
                    distance_dict = sorted(temp_dict, key=temp_dict.get)
                    for column in reversed_missing_dict[i]:
                        temp_iter = (
                            stored_row.iat[sub_index, stored_row.columns.get_loc(column)]
                            for sub_index in distance_dict
                            if sub_index not in missing_dict[column]
                        )
                        temp_array = list(islice(temp_iter, k))
                        if column in self.continuous_attribute_names:
                            if len(temp_array) > 0:
                                self.raw.at[i, column] = sum(temp_array) / len(temp_array)
                        elif column in self.discrete_attribute_names:
                            if len(temp_array) > 0:
                                self.raw.at[i, column] = np.median(temp_array)
                        else:
                            if len(temp_array) > 0:
                                self.raw.at[i, column] = stats.mode(temp_array)[0]
            self.raw.to_csv(str(not_used)+'temp.csv')
