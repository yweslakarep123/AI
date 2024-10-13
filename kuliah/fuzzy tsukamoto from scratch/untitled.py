class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

class DecisionTree:
    def __init__(self, max_depth=100, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None

    def fit(self, X, y):
        self.root = self._grow_tree(X, y)

    def _grow_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape
        n_classes = len(np.unique(y))

        if (depth >= self.max_depth or n_samples < self.min_samples_split or n_classes == 1):
            return Node(value=self._most_common_label(y))

        feat_idxs = np.random.choice(n_features, n_features, replace=False)
        best_feature, best_threshold = self._best_split(X, y, feat_idxs)

        left_idxs = X[:, best_feature] < best_threshold
        right_idxs = ~left_idxs

        left = self._grow_tree(X[left_idxs], y[left_idxs], depth + 1)
        right = self._grow_tree(X[right_idxs], y[right_idxs], depth + 1)

        return Node(best_feature, best_threshold, left, right)

    def _best_split(self, X, y, feat_idxs):
        best_gain, split_idx, split_threshold = -1, None, None

        for feat_idx in feat_idxs:
            X_column = X[:, feat_idx]
            thresholds = np.unique(X_column)

            for threshold in thresholds:
                gain = self._information_gain(y, X_column, threshold)
                if gain > best_gain:
                    best_gain, split_idx, split_threshold = gain, feat_idx, threshold

        return split_idx, split_threshold

    def _information_gain(self, y, X_column, threshold):
        parent_entropy = self._entropy(y)
        left_idxs = X_column < threshold
        right_idxs = ~left_idxs

        if len(y[left_idxs]) == 0 or len(y[right_idxs]) == 0:
            return 0

        n = len(y)
        n_l, n_r = len(y[left_idxs]), len(y[right_idxs])
        e_l, e_r = self._entropy(y[left_idxs]), self._entropy(y[right_idxs])
        child_entropy = (n_l / n) * e_l + (n_r / n) * e_r

        return parent_entropy - child_entropy

    def _entropy(self, y):
        hist = np.bincount(y)
        ps = hist / len(y)
        return -np.sum(p * np.log2(p) for p in ps if p > 0)

    def _most_common_label(self, y):
        return Counter(y).most_common(1)[0][0]

    def predict(self, X):
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def _traverse_tree(self, x, node):
        if node.value is not None:
            return node.value
        return self._traverse_tree(x, node.left if x[node.feature] < node.threshold else node.right)

# Load and prepare the data
def load_data(file_path):
    data = pd.read_csv(file_path)
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values
    unique_classes = np.unique(y)
    class_to_int = {cls: i for i, cls in enumerate(unique_classes)}
    y = np.array([class_to_int[cls] for cls in y])
    return X, y

# Split the data into training and testing sets
def train_test_split(X, y, train_size=0.8):
    np.random.seed(42)
    indices = np.random.permutation(len(X))
    train_size = int(train_size * len(X))
    return X[indices[:train_size]], X[indices[train_size:]], y[indices[:train_size]], y[indices[train_size:]]

# Calculate precision and recall
def calculate_metrics(y_true, y_pred):
    classes = np.unique(y_true)
    precision = {}
    recall = {}
    accuracy = np.sum(y_pred == y_true) / len(y_true)
    for c in classes:
        true_positives = np.sum((y_pred == c) & (y_true == c))
        predicted_positives = np.sum(y_pred == c)
        actual_positives = np.sum(y_true == c)

        precision[c] = true_positives / predicted_positives if predicted_positives > 0 else 0
        recall[c] = true_positives / actual_positives if actual_positives > 0 else 0

        avg_precision = np.mean(list(precision.values()))
        avg_recall = np.mean(list(recall.values()))

    return accuracy, avg_precision, avg_recall

# Main execution
X, y = load_data('C:/Users/Thinkpad/Documents/ml_prak/Tugas_MLprak_tm6/wine_dataset.csv')
X_train, X_test, y_train, y_test = train_test_split(X, y)

# Train the Decision Tree
model = DecisionTree()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy, average_precision, average_recall = calculate_metrics(y_test, y_pred)

print(f"Accuracy: {accuracy:.4f}")
print(f"Average Precision: {average_precision:.4f}")
print(f"Average Recall: {average_recall:.4f}")