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
    ]},
    {'name': 'Hypothesis', 'applications': [
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
            'description': 'Perform regression with OLS estimated linear model.',
            'opened_link': '/algo_linear_regression/add'
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
    ]},
    {'name': 'Advanced Applications', 'applications': [
        {
            'name': '',
            'description': '',
            'opened_link': ''
        },
    ]},
]
