from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris

if __name__ == '__main__':
    iris = load_iris()
    features, labels = iris.data, iris.target

    train_features, test_features, train_labels, test_labels = train_test_split(features, labels,
                                                                                test_size=0.33, random_state=0)
    clf = DecisionTreeClassifier(criterion='gini')
    clf.fit(train_features, train_labels)

    test_pred = clf.predict(test_features)
    print('Cart分类树准确率为: %.4lf' % (accuracy_score(test_labels, test_pred)))
