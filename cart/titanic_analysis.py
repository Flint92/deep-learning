import pandas as pd
import numpy as np

from sklearn.feature_extraction import DictVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score

if __name__ == '__main__':
    train_data = pd.read_csv('titanic_train.csv')
    test_data = pd.read_csv('titanic_test.csv')

    train_data['Age'].fillna(train_data['Age'].mean(), inplace=True)
    test_data['Age'].fillna(test_data['Age'].mean(), inplace=True)

    train_data['Fare'].fillna(train_data['Fare'].mean(), inplace=True)
    test_data['Fare'].fillna(test_data['Fare'].mean(), inplace=True)

    train_data['Embarked'].fillna('S', inplace=True)
    test_data['Embarked'].fillna('S', inplace=True)

    # 特征选择
    features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
    train_features = train_data[features]
    train_labels = train_data['Survived']

    test_features = test_data[features]

    dvec = DictVectorizer(sparse=False)
    train_features = dvec.fit_transform(train_features.to_dict(orient='records'))
    print(dvec.feature_names_)

    clf = DecisionTreeClassifier(criterion='entropy') # ID3
    clf.fit(train_features, train_labels)

    test_features = dvec.transform(test_features.to_dict(orient='records'))
    pred_labels = clf.predict(test_features)

    # 得到决策树准确率
    acc_decision_tree = round(clf.score(train_features, train_labels), 6)
    print(u'score准确率为 %.4lf' % acc_decision_tree)

    # 交叉验证
    print(u'cross_val_score准确率为 %.4lf' % np.mean(cross_val_score(clf, train_features, train_labels, cv=10)))