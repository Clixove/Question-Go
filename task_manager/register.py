algorithm_registry = [
    {'name': 'Data pre-processing', 'applications': [
        {
            'name': 'Cross-sectional data',
            'description': 'Process cross-sectional data: dropping, filling missing values, converting type, '
                           'encoding, and basic maths operations.',
            'opened_link': '/pre_cross_sectional/add'
        },
        {
            'name': 'Time series data',
            'description': 'Converting records from a 2D table to a multi-dimension time series.',
            'opened_link': '/pre_ts/add'
        },
        {
            'name': 'Normalization',
            'description': 'Perform standardization and 0~1 zipping.',
            'opened_link': '/pre_norm/add',
        },
        {
            'name': 'PCA',
            'description': 'Principal component analysis, also a decomposition method.',
            'opened_link': '/algo_pca/add',
        },
        {
            'name': 'Resampling',
            'description': 'Resample for the imbalanced classification dataset.',
            'opened_link': '/pre_resampling/add',
        },
    ]},
    {'name': 'Classification', 'applications': [
        {
            'name': 'Random forest classifier',
            'description': 'Perform classification with random forest algorithm. Bayes hyper-parameters search is '
                           'applied.',
            'opened_link': '/algo_rf_classifier/add'
        },
        {
            'name': 'SVM classifier',
            'description': 'Perform classification with support vector machine algorithm. Bayes hyper-parameters '
                           'search is applied.',
            'opened_link': '/algo_svm_classifier/add'
        },
        {
            'name': 'One-class SVM',
            'description': 'Train the model with all normal samples, to detect abnormal samples.',
            'opened_link': '/algo_one_class_svm/add'
        },
        {
            'name': 'Logistic regression',
            'description': 'Perform classification with L1, L2 regularized logistic model. Bayes hyper-parameters '
                           'search is applied.',
            'opened_link': '/algo_logistic_regression/add'
        },
    ]},
    {'name': 'Regression', 'applications': [
        {
            'name': 'Linear regression',
            'description': 'Perform regression with linear model.',
            'opened_link': '/algo_linear_regression/add'
        },
        {
            'name': 'Elastic net',
            'description': 'Perform regressin with L1, L2 regularized linear model.',
            'opened_link': '/algo_elastic_net/add'
        },
        {
            'name': 'Random forest regression',
            'description': 'Perform regression with random forest algorithm. Bayes hyper-parameters search is applied.',
            'opened_link': '/algo_rf_regressor/add'
        },
        {
            'name': 'SVM regression',
            'description': 'Perform regression with support vector machine algorithm. Bayes hyper-parameters search is '
                           'applied.',
            'opened_link': '/algo_svm_regressor/add'
        },
    ]},
    {'name': 'Prediction', 'applications': [
    ]},
    {'name': 'Clustering', 'applications': [
        {
            'name': 'DBSCAN',
            'description': 'Clustering by density when the number of clusters isn\'t available.',
            'opened_link': '/algo_dbscan/add'
        },
        {
            'name': 'K Means',
            'description': 'Clustering by distance when the number of clusters is given.',
            'opened_link': '/algo_kmeans/add'
        },
    ]},
]
