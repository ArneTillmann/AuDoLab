from sklearn.svm import OneClassSVM
from pandas import DataFrame


class One_Class_SVM:
    def __init__(self):
        3 + 4

    @staticmethod
    def choose_classifier(df, classifier, i):
        """
        returns dataframe where documents that are classified to target class
        have 1, otherwise, 0


        :param df: dataframe of target documents
        :type df: pd.Dataframe
        :param classifier: list of all possible o-svm classifiers
        :type classifier: list
        :param i: index of which classifier is chosen/preferred
        :type i: int

        :return: documents that are classified as belonging to target
        :rtype: pd.dataframe
            class by o-svm
        """
        return df.iloc[classifier.index[classifier.iloc[:, i] == 1].tolist()]

    @staticmethod
    def classification(
        training,
        predicting,
        nus,
        quality_train=0.85,
        min_pred=0.05,
        max_pred=0.2,
        gamma="auto",
        kernel="rbf",
    ):
        """[summary]

        Args:
            training (DataFrame): training dataset of preprocessed documents
            predicting (DataFrame): target dataset of preprccessed documents
            nus (list of floats): hyperparameters over which are looped. For each nu the classifier is trained
            quality_train (float, optional): percentage of training data that seems to
                                    belong to target class. Default: 0.85. Defaults to 0.85.
            min_pred (float, optional): percentage of target data that has to be at
                               least classified as belonging to target class
            for classifier to be considered. Default: 0.0. Defaults to 0.05.
            max_pred (float, optional): percentage of target class that is maximally
                               allowed to be classified as belonging to
            target class for classifier to be considered.. Defaults to 0.2.
            gamma (str, optional): Hyperparamter of O-SVM. Defaults to "auto".
            kernel (str, optional): Kernel function used in O_SVM. Defaults to "rbf".

        Returns:
            pd.DataFrame: DataFrame with stored classifiers that fulfill
                conditions
        """

        df = DataFrame()
        for i in nus:

            svm = OneClassSVM(nu=i, gamma=gamma, kernel=kernel)
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
