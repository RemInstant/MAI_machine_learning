import numpy as np
from collections import Counter


def find_best_split(feature_vector, target_vector):
  feature_vector = np.array(feature_vector)
  target_vector = np.array(target_vector)
  n = len(feature_vector)

  sorted_ind = np.argsort(feature_vector)
  feature_vector = feature_vector[sorted_ind]
  target_vector = target_vector[sorted_ind]


  unique_indices = np.where(feature_vector[1:] != feature_vector[:-1])[0]
  if (len(unique_indices) == 0):
    return None, None, None, None

  all_pref_cnt_left = np.arange(1, n + 1)
  p1_pref_cnt_left = np.cumsum(target_vector)
  p0_pref_cnt_left = all_pref_cnt_left - p1_pref_cnt_left

  p1_cnt = p1_pref_cnt_left[-1]
  p0_cnt = p0_pref_cnt_left[-1]

  all_pref_cnt_left = all_pref_cnt_left[:-1]
  p1_pref_cnt_left = p1_pref_cnt_left[:-1]
  p0_pref_cnt_left = p0_pref_cnt_left[:-1]

  all_pref_cnt_right = n - all_pref_cnt_left
  p1_pref_cnt_right = p1_cnt - p1_pref_cnt_left
  p0_pref_cnt_right = p0_cnt - p0_pref_cnt_left

  p1_pref_rate_left = p1_pref_cnt_left / all_pref_cnt_left
  p0_pref_rate_left = p0_pref_cnt_left / all_pref_cnt_left
  p1_pref_rate_right = p1_pref_cnt_right / all_pref_cnt_right
  p0_pref_rate_right = p0_pref_cnt_right / all_pref_cnt_right

  h_pref_left = 1 - p1_pref_rate_left**2 - p0_pref_rate_left**2
  h_pref_right = 1 - p1_pref_rate_right**2 - p0_pref_rate_right**2

  thresholds = (feature_vector[1:] + feature_vector[:-1]) / 2
  ginis = -all_pref_cnt_left / n * h_pref_left - all_pref_cnt_right / n * h_pref_right

  ginis = ginis[unique_indices]
  thresholds = thresholds[unique_indices]
  best_ind = np.argmax(ginis)

  return thresholds, ginis, thresholds[best_ind], ginis[best_ind]

find_best_split([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [1, 0, 0, 0, 1, 1, 0, 1, 1, 1])


class DecisionTree:
  def __init__(self, feature_types, max_depth=None, min_samples_split=None, min_samples_leaf=None):
    if np.any(list(map(lambda x: x != 'real' and x != 'categorical', feature_types))):
      raise ValueError('There is unknown feature type')

    self._root = {}
    self._depth = 0
    self._feature_types = feature_types
    self._max_depth = max_depth
    self._min_samples_split = min_samples_split
    self._min_samples_leaf = min_samples_leaf

  def _fit_node(self, sub_X, sub_y, node):
    if np.all(sub_y == sub_y[0]): # fix: != -> ==
      node['type'] = 'terminal'
      node['class'] = sub_y[0]
      return

    feature_best, threshold_best, gini_best, split = None, None, None, None
    for feature in range(1, sub_X.shape[1]):
      feature_type = self._feature_types[feature]
      categories_map = {}

      if feature_type == 'real':
        feature_vector = sub_X[:, feature]
      elif feature_type == 'categorical':
        counts = Counter(sub_X[:, feature])
        clicks = Counter(sub_X[sub_y == 1, feature])
        ratio = {}
        for key, current_count in counts.items():
          if key in clicks:
            current_click = clicks[key]
          else:
            current_click = 0
          ratio[key] = current_click / current_count # fix: swapped operands
        sorted_ratio = sorted(ratio.items(), key=lambda x: x[1])
        sorted_categories = list(map(lambda x: x[0], sorted_ratio)) # fix: index 0 (category) has to be exracted, not index 1
        categories_map = dict(zip(sorted_categories, list(range(len(sorted_categories)))))
        feature_vector = np.array(list(map(lambda x: categories_map[x], sub_X[:, feature])))
      else:
        raise ValueError

      _, _, threshold, gini = find_best_split(feature_vector, sub_y)
      if gini is None:
        continue

      if gini_best is None or gini > gini_best:
        feature_best = feature
        gini_best = gini
        left_indices = feature_vector < threshold
        right_indices = feature_vector > threshold

        if feature_type == 'real':
          threshold_best = threshold
        elif feature_type == 'categorical':
          threshold_best = list(map(lambda x: x[0],
                        filter(lambda x: x[1] < threshold, categories_map.items())))
        else:
          raise ValueError

    if feature_best is None:
      node['type'] = 'terminal'
      node['class'] = Counter(sub_y).most_common(1)[0][0] # fix: retrieved class from array
      return

    node['type'] = 'nonterminal'

    node['feature_split'] = feature_best
    if self._feature_types[feature_best] == 'real':
      node['threshold'] = threshold_best
    elif self._feature_types[feature_best] == 'categorical':
      node['categories_split'] = threshold_best
    else:
      raise ValueError
    node['left_child'], node['right_child'] = {}, {}
    node['left_child']['depth'] = node['depth'] + 1
    node['right_child']['depth'] = node['depth'] + 1
    self._depth = max(self._depth, node['depth'] + 1)
    self._fit_node(sub_X[left_indices], sub_y[left_indices], node['left_child'])
    self._fit_node(sub_X[right_indices], sub_y[right_indices], node['right_child']) # fix: put RIGHT y elemets

  def _predict_node(self, x, node):
    if node['type'] == 'terminal':
      return node['class']

    feature_split = node['feature_split']

    if self._feature_types[feature_split] == 'real':
      go_left = x[feature_split] < node['threshold']
    elif self._feature_types[feature_split] == 'categorical':
      go_left = x[feature_split] in node['categories_split']
    else:
      raise ValueError

    if go_left:
      return self._predict_node(x, node['left_child'])
    else:
      return self._predict_node(x, node['right_child'])

  def fit(self, X, y):
    X = np.array(X)
    y = np.array(y)
    self._root = { 'depth': 1 }
    self._fit_node(X, y, self._root)

  def predict(self, X):
    X = np.array(X)
    predicted = []
    for x in X:
      predicted.append(self._predict_node(x, self._root))
    return np.array(predicted)
