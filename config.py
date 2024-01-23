import argparse

parser = argparse.ArgumentParser(description='Detect AI generated text')

parser.add_argument('--iter_u', type=int, default=100)
parser.add_argument('--test_none', type=str, default="None")

# MultinomialNB
parser.add_argument('--nb_alpha', type=float, default=0.1)

# SGDClassifier
parser.add_argument('--sgd_iter', type=int, default=1)
parser.add_argument('--sgd_penalty', type=str, default='l2')
parser.add_argument('--sgd_alpha', type=float, default=0.0001)
parser.add_argument('--sgd_tol', type=float, default=0.0001)
parser.add_argument('--sgd_learning_rate', type=str, default='optimal')
parser.add_argument('--sgd_eta0', type=float, default=0.001)
parser.add_argument('--sgd_power_t', type=float, default=0.5)
parser.add_argument('--sgd_early_stopping', type=bool, default=False)
parser.add_argument('--sgd_warm_start', type=bool, default=False)

# LGBMClassifier
parser.add_argument('--lgbm_boosting_type', type=str, default='gbdt')
parser.add_argument('--lgbm_num_leaves', type=int, default=31)
parser.add_argument('--lgbm_max_depth', type=int, default=-1)
parser.add_argument('--lgbm_learning_rate', type=float, default=0.1)
parser.add_argument('--lgbm_n_estimators', type=int, default=100)
parser.add_argument('--lgbm_subsample_for_bin', type=int, default=200000)
parser.add_argument('--lgbm_colsample_bytree', type=float, default=1.0)
parser.add_argument('--lgbm_colsample_bynode', type=float, default=1.0)
parser.add_argument('--lgbm_reg_alpha', type=float, default=0.0)
parser.add_argument('--lgbm_reg_lambda', type=float, default=0.0)

# CatBoostClassifier
parser.add_argument('--catboost_iter', type=int, default=1)
parser.add_argument('--catboost_learning_rate', type=float, default=0.1)
parser.add_argument('--catboost_subsample', type=float, default=1.0)
parser.add_argument('--catboost_l2_leaf_reg', type=float, default=3.0)
parser.add_argument('--catboost_random_strength', type=float, default=1.0)
parser.add_argument('--catboost_bagging_temperature', type=float, default=1.0)

# VotingClassifier
parser.add_argument('--voting_weight_nb', type=float, default=0.2)
parser.add_argument('--voting_weight_sgd', type=float, default=0.31)
parser.add_argument('--voting_weight_lgbm', type=float, default=0.31)
parser.add_argument('--voting_weight_catboost', type=float, default=0.46)

args = parser.parse_args()
