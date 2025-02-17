from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.tree import DecisionTreeRegressor


if __name__ == '__main__':
    housing = fetch_openml(name='boston', version=1, as_frame=True, parser='auto')
    features, prices = housing.data, housing.target
    print(features)
    print(prices)

    train_features, test_features, train_prices, test_prices = train_test_split(
        features, prices, test_size=0.33)

    clf = DecisionTreeRegressor()
    clf.fit(train_features, train_prices)

    test_pred = clf.predict(test_features)
    print('决策树回归预测价格均方误差为: %.4lf' % (mean_squared_error(test_prices, test_pred)))
    print('决策树回归预测价格平均绝对误差为: %.4lf' % (mean_absolute_error(test_prices, test_pred)))
    print('决策树回归预测价格R2值为: %.4lf' % (r2_score(test_prices, test_pred)))

