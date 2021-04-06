from sklearn.metrics import roc_auc_score
from sklearn.svm import OneClassSVM
import pandas as pd


class One_Class_SVM:
    """
    Keywordscraper
    """

    def __init__(self):
        3+4

    @staticmethod
    def choose_classifier(df, classifier, i):
        return pd.DataFrame(df["tokens"][classifier.index[classifier.iloc[:, i] == 1].tolist()])

    @staticmethod
    def classification(training, predicting, nus, quality_train, min_pred, max_pred):
        df = pd.DataFrame()
        for i in nus:

            svm = OneClassSVM(nu=i, gamma="auto", kernel="rbf")
            # fit the model for each kernel
            clf = svm.fit(training)

            train = clf.predict(training)

            if sum(train[train == 1]) >= round(quality_train * len(train)):

                prediction = clf.predict(predicting)

                if (
                    sum(prediction[prediction == 1])
                    >= round(min_pred * (predicting.shape[0]))
                ) and (
                    sum(prediction[prediction == 1])
                    <= round(max_pred * (predicting.shape[0]))
                ):
                    print(
                        "nu:",
                        str(i),
                        "data predicted:",
                        sum(prediction[prediction == 1]),
                        "training_data predicted:",
                        sum(train[train == 1]),
                    )

                    df["prediction: nu: ", str(i)] = prediction

        return df
