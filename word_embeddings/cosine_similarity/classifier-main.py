import os
import warnings

import pandas as pd
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import export_graphviz, DecisionTreeClassifier

warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')

OUTPUT_DIR = os.path.join('..', 'outputs')
DATA_DIR = os.path.join('..', 'data')


def classify(model, params):
    gcv = GridSearchCV(model, params, cv=10, scoring="accuracy")
    gcv.fit(X_train, y_train)
    model = gcv.best_estimator_
    y_hat = model.predict(X_test)

    print("Printing results for", model.__class__.__name__)
    print("With best params", gcv.best_params_)
    print("Train accuracy", accuracy_score(y_train, model.predict(X_train)))
    print("Test accuracy", accuracy_score(y_test, y_hat))
    print(metrics.classification_report(y_test, y_hat))

    feature_names = [name for name in X_train.columns]
    print('The following features are important:')
    most_important_features_indices = [(i, mvf) for i, mvf in enumerate(model.feature_importances_) if mvf > 0]

    for i, mvf in most_important_features_indices:
        print('feature {0}: {1}: {2:.5f}%'.format(i, data.columns[i], mvf * 100))

    if isinstance(model, DecisionTreeClassifier):
        with open(os.path.join(OUTPUT_DIR, 'tree.txt'), "w") as f:
            export_graphviz(model, out_file=f, feature_names=feature_names)


if __name__ == '__main__':
    df = pd.read_csv(os.path.join(DATA_DIR, 'features_all.csv'))
    column_names = ['noun',
                    'verb',
                    'adjective',
                    'adverb',
                    'all',
                    'positive_count',
                    'negative_count',
                    'cos_sim'
                    ]
    data = pd.DataFrame()

    for col in column_names:
        regex = 'q_[1-9]+_{}.*'.format(col)
        cols = df.filter(regex=regex, axis=1)
        data = pd.concat([data, cols], axis=1)
    data = pd.concat([data, df['label']], axis=1)
    data.dropna(inplace=True)

    y = data['label'].replace({'control': 0, 'patient': 1})
    X = data.drop('label', axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    classify(DecisionTreeClassifier(),
             params={"criterion": ["gini", "entropy"],
                     "max_depth": [3, 4, 5, 6, 7, 8, 9, 10, 15]}
             )
    classify(RandomForestClassifier(),
             params={"criterion": ["gini", "entropy"],
                     "max_depth": [3, 4, 5, 6, 7, 8, 9, 10, 15],
                     "n_estimators": [2, 3, 4, 5, 6, 7, 8, 9, 10, 15]}
             )
