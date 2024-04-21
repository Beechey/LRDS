import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
import time
from sklearn.preprocessing import normalize
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Lasso
from sklearn.decomposition import PCA

# from __future__ import print_function
import numpy as np
from sklearn import linear_model
from genetic_selection import GeneticSelectionCV

# so scripts from other folders can import this file
dir_path = os.path.abspath(os.path.dirname(os.path.realpath(__file__)))


def clean_dataset(df):
    assert isinstance(df, pd.DataFrame), "df needs to be a pd.DataFrame"

    df = df.replace('NaN', np.nan)
    df = df.replace('Infinity', np.nan)
    df = df.replace('infinity', np.nan)
    df = df.replace('inf', np.nan)
    df = df.replace(np.inf, np.nan)
    df = df.replace('', np.nan)
    df = df.dropna(axis=0)

    df.dropna(inplace=True)
    indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(1)

    return df[indices_to_keep].astype(np.float64)


def get_data():
    '''
    1. Opens a CSV dataset file and add to a dataframe\n
    2. Variables for raw, normalised and Labels  (rX, nX, Y, respectively)\n
    3. Manually add feature names (This could be automated if CSV file has Header/titles)\n
    '''

    dir_path = "datasets/"
    technique = "all_features"

    print("Pulling in the dataset... ", end="", flush=True)
    df = pd.read_csv(dir_path + '1502_combined_attacks.csv', low_memory=False)
    # df = pd.read_csv(dir_path + '1502_goldeneye.csv', low_memory=False)
    # df = pd.read_csv(dir_path + '1502_slowloris.csv', low_memory=False)

    # df = pd.read_csv(dir_path + '1402_bf_combined_2_features.csv', low_memory=False)
    # df = pd.read_csv(dir_path + '1402_bf_combined_test.csv', low_memory=False)
    # df = pd.read_csv(dir_path + '1402_bf_combined.csv', low_memory=False)
    # df = pd.read_csv(dir_path + '1402_ssh_bf.csv', low_memory=False)

    # df = pd.read_csv(dir_path + 'heart_data.csv', low_memory=False)
    # df = pd.read_csv(dir_path + 'attack02_dataset3.csv', low_memory=False)

    # hightlighted feature combos

    # df = df[['ACK Flag Cnt', 'Init Fwd Win Byts', 'Label']]
    # df = df[['Fwd Pkt Len Min', 'Init Bwd Win Byts', 'Label']]
    # df = df[['Fwd Pkts/s', 'Init Fwd Win Byts', 'Label']]
    # df = df[['Pkt Len Max', 'Init Bwd Win Byts', 'Label']]
    # df = df[['Pkt Len Max', 'Init Fwd Win Byts', 'Label']]
    # df = df[['Pkt Len Mean', 'Idle Min', 'Label']]
    # df = df[['Pkt Len Min', 'Init Bwd Win Byts', 'Label']]
    # df = df[['Pkt Len Mean', 'Bwd Seg Size Avg', 'Label']]
    

    # -------- GoldenEye

    # LRDS Goldeneye top 5
    # technique = "lr-ds"
    # df = df[['PSH Flag Cnt', 'Fwd Seg Size Min', 'Protocol', 'Bwd Pkt Len Std', 'Pkt Len Var', 'Label']]
    
    # ANOVA Goldeneye top 5
    # technique = "anova"
    # df = df[['Fwd Seg Size Min', 'Init Fwd Win Byts', 'Fwd Pkt Len Std', 'Bwd Pkt Len Std', 'Protocol', 'Label']]

    # ExtraTreesClassifier Importance Goldeneye top 5
    # technique = "extra_trees_classifier"
    # df = df[['Fwd Seg Size Min', 'Init Fwd Win Byts', 'Bwd Pkt Len Max', 'Bwd Pkt Len Std', 'Fwd IAT Min', 'Label']]

    # DecisionTreeClassifier Importance Goldeneye top 5
    # technique = "decision_tree_classifier"
    # df = df[['Fwd Seg Size Min', 'Dst Port', 'Init Bwd Win Byts', 'URG Flag Cnt', 'Init Fwd Win Byts', 'Label']]

    # LR Permutations Importance Goldeneye top 5
    # technique = "lr_permutations_importance"
    # df = df[['Fwd IAT Max', 'Flow Duration', 'Fwd IAT Tot', 'Flow IAT Max', 'Bwd IAT Mean', 'Label']]

    # Random Forest Classifier Goldeneye top 5
    # technique = "random_forest_classifier"
    # df = df[['Fwd Seg Size Min', 'Init Fwd Win Byts', 'Fwd Header Len', 'Flow Duration', 'Fwd IAT Max', 'Label']]

    # Lasso Goldeneye top 5
    # technique = "lasso"
    # df = df[['Bwd Pkt Len Std', 'Fwd Pkt Len Std', 'Pkt Len Max', 'Fwd Pkt Len Max', 'Init Fwd Win Byts', 'Label']]
    

    # -------- SLOWLORIS

    # LRDS Slowloris top 5
    # technique = "lr-ds"
    # df = df[['Fwd Seg Size Min', 'PSH Flag Cnt', 'Dst Port', 'Protocol', 'ACK Flag Cnt', 'Label']]

    # DecisionTreeClassifier Importance Slowloris top 5
    # technique = "decision_tree_classifier"
    # df = df[['Bwd IAT Max', 'Dst Port', 'Fwd Seg Size Min', 'Init Fwd Win Byts', 'Down/Up Ratio', 'Label']]

    # -------- COMBINED

    # LRDS Combined top 5
    # technique = "lr-ds"
    # df = df[['PSH Flag Cnt', 'Dst Port', 'Fwd Seg Size Min', 'Pkt Len Std', 'Bwd Pkt Len Std', 'Label']] # all labels
    # df = df[[   'PSH Flag Cnt', 'Dst Port', 'Fwd Seg Size Min', 'Pkt Len Std', 'Bwd Pkt Len Std', 'Protocol', 'Pkt Len Var', 'ACK Flag Cnt', 'Fwd Pkt Len Std', 'Fwd Pkt Len Max', 
    #             'Pkt Len Max', 'Bwd Pkt Len Max', 'Fwd IAT Max', 'Idle Max', 'Bwd IAT Tot', 'Bwd Pkt Len Mean', 'Bwd Seg Size Avg', 'Fwd IAT Min', 'Bwd IAT Std', 'Bwd Pkt Len Min', 
    #             'Fwd IAT Mean', 'Flow IAT Mean', 'Fwd Header Len', 'Init Fwd Win Byts', 'Flow IAT Min', 'Flow IAT Std', 'Bwd IAT Max', 'Fwd Act Data Pkts', 'Bwd IAT Mean', 'Flow IAT Max', 
    #             'Active Std', 'Active Min', 'RST Flag Cnt', 'ECE Flag Cnt', 'Pkt Len Min', 'Bwd IAT Min', 'Active Mean', 'Fwd Pkt Len Min', 'FIN Flag Cnt', 'Flow Pkts/s',
    #             'Fwd Pkts/s', 'Bwd Pkts/s', 'Fwd Pkt Len Mean', 'Fwd Seg Size Avg', 'Bwd PSH Flags', 'Fwd URG Flags', 'Bwd URG Flags', 'CWE Flag Count', 'Fwd Byts/b Avg', 'Fwd Pkts/b Avg',
    #             'Fwd Blk Rate Avg', 'Bwd Byts/b Avg', 'Bwd Pkts/b Avg', 'Bwd Blk Rate Avg', 'Pkt Size Avg', 'Fwd IAT Tot', 'Init Bwd Win Byts', 'Down/Up Ratio', 'TotLen Fwd Pkts', 'Subflow Fwd Byts',
    #             'Flow Byts/s', 'Tot Fwd Pkts', 'Subflow Fwd Pkts', 'TotLen Bwd Pkts', 'Subflow Bwd Byts', 'Bwd Header Len', 'Fwd PSH Flags', 'SYN Flag Cnt', 'Idle Std', 'Fwd IAT Std', 'Label']] # all labels
    # print(df.shape)

    # sys.exit()
    # df = df[['Fwd Seg Size Min', 'Fwd Act Data Pkts', 'Fwd Pkt Len Max', 'Bwd Pkts/s', 'Bwd Pkt Len Max', 'Label']] # positive labels only

    # LRDS Combined top 10
    # technique = "lr-ds"
    # df = df[['PSH Flag Cnt', 'Dst Port', 'Fwd Seg Size Min', 'Pkt Len Std', 'Bwd Pkt Len Std', 'Protocol', 'Pkt Len Var', 'ACK Flag Cnt', 'Fwd Pkt Len Std', 'Fwd Pkt Len Max', 'Label']] # all labels


    # ANOVA Combined top 5
    # technique = "anova"
    # df = df[['Fwd Seg Size Min', 'Bwd IAT Mean', 'Init Fwd Win Byts', 'Bwd IAT Min', 'Flow IAT Std', 'Label']]

    # Permutation Importance Combined top 5
    # technique = "lr_permutation_importance"
    # df = df[['Flow Duration', 'Fwd IAT Max', 'Fwd IAT Tot', 'Bwd IAT Mean', 'Bwd IAT Std', 'Label']]

    # Decision Trees Combined top 5
    # technique = "decision_trees"
    # df = df[['Fwd Seg Size Min', 'Dst Port', 'Init Fwd Win Byts', 'Down/Up Ratio', 'Fwd IAT Max', 'Label']]

    # Extra Trees Combined top 5
    # technique = "extra_trees"
    # df = df[['Fwd Seg Size Min', 'Init Fwd Win Byts', 'Fwd IAT Min', 'Bwd IAT Mean', 'Bwd Pkt Len Max', 'Label']]

    # Random Forest Combined top 5
    # technique = "random_forest"
    # df = df[['Fwd Seg Size Min', 'Init Fwd Win Byts', 'Dst Port', 'Fwd Header Len', 'Fwd Pkts/s', 'Label']]

    # Lasso Combined top 5
    # technique = "lasso"
    # df = df[['Bwd Pkt Len Std', 'Fwd Pkt Len Max', 'Fwd Pkt Len Std', 'Pkt Len Max', 'Init Fwd Win Byts', 'Label']]

    # PCA Combined top 5
    technique = "pca"
    # df = df[['Subflow Fwd Pkts', 'Subflow Fwd Byts', 'Subflow Bwd Pkts', 'Subflow Bwd Byts', 'Init Fwd Win Byts', 'Label']]
    df = df[['Flow Duration', 'Fwd IAT Tot', 'Bwd IAT Tot', 'Flow IAT Max', 'Fwd IAT Max', 'Label']]

    # ------------ SSH BF

    # LRDS uncertainty SSH BF top 5
    # technique = "lr-ds"
    # df = df[['Fwd Seg Size Min', 'Bwd Pkt Len Max', 'PSH Flag Cnt', 'Fwd Pkt Len Max', 'Pkt Len Max', 'Label']]   # <--- mean


    # ------------ BF Combined
    # LRDS uncertainty combined BF top 5
    # technique = "lr-ds"
    # df = df[['Fwd Seg Size Min', 'Bwd Pkts/s', 'Fwd Act Data Pkts', 'Bwd Pkt Len Max', 'Flow Pkts/s', 'Label']]   # <--- mean
    # df = df[['Fwd Seg Size Min', 'Bwd Pkts/s', 'Fwd Act Data Pkts', 'Flow Pkts/s', 'Fwd Pkt Len Max', 'Label']]   # <--- median
    # df = df[['Fwd Seg Size Min', 'Fwd Pkt Len Max', 'Fwd Act Data Pkts', 'Bwd Pkts/s', 'Bwd Pkt Len Max', 'Label']]   # <--- only positive class

    # LRDS new methodology BF top 5
    # technique = "lr-ds_new"
    # df = df[['Fwd Seg Size Min', 'Init Fwd Win Byts', 'Bwd Pkts/s', 'Flow Pkts/s', 'Fwd Pkt Len Mean', 'Label']]

    # Decision Tree BF top 5
    # technique = "decision_trees"
    # df = df[['Fwd Seg Size Min', 'Dst Port', 'Flow Pkts/s', 'Init Fwd Win Byts', 'Protocol', 'Label']]

    # -------- South African Heart Disease

    # df = df[['ldl', 'age', 'chd']]
    # df = df[['rssi_jitter', 'data_rate_synth', 'y']]
    # df.columns = ['RSSI', 'Data Rate (Mbps)', 'y']
    # df = df[['rssi', 'seq_diff', 'y']]
    print("Done!")

    print("Cleaning up dataset... ", end="", flush=True)
    df = clean_dataset(df)
    print("Done!")

    # print(df.shape)

    fs_x = df.iloc[:, :-1]
    fs_y = df.iloc[:, -1]

    # start = time.process_time()
    
    #apply SelectKBest class to extract top 5 best features
    # bestfeatures = SelectKBest(score_func=f_classif, k=5)
    # fit = bestfeatures.fit(fs_x,fs_y)
    # dfscores = pd.DataFrame(fit.scores_)
    # dfcolumns = pd.DataFrame(fs_x.columns)
    # #concat two dataframes for better visualization 
    # featureScores = pd.concat([dfcolumns,dfscores],axis=1)
    # featureScores.columns = ['Feature','Score']  #naming the dataframe columns
    # print("ANOVA")
    # print(featureScores.nlargest(5,'Score'))  #print 5 best features

    # anova = time.process_time() - start
    

    # start = time.process_time()

    # #LR Permutations Importance
    # model = LogisticRegression().fit(fs_x, fs_y)
    # fit = permutation_importance(model, fs_x, fs_y, n_repeats=2, random_state=0)

    # # fig, ax = plt.subplots()
    # # sorted_idx = fit.importances_mean.argsort()
    # # ax.boxplot(fit.importances[sorted_idx].T,
    # #         vert=False, labels=range(fs_x.shape[1]))
    # # ax.set_title("Permutation Importance of each feature")
    # # ax.set_ylabel("Features")
    # # ax.set_xlabel("Importance")
    # # fig.tight_layout()
    # # plt.show()

    # dfscores = pd.DataFrame(fit.importances_mean)
    # dfcolumns = pd.DataFrame(fs_x.columns)
    # #concat two dataframes for better visualization 
    # featureScores = pd.concat([dfcolumns,dfscores],axis=1)
    # featureScores.columns = ['Feature','Importance']  #naming the dataframe columns
    # print("Permutation Importance")
    # print(featureScores.nlargest(5,'Importance'))  #print 5 best features


    # lr_pi = time.process_time() - start
    

    # start = time.process_time()

    # # Decision Tree Classifier
    # dt = DecisionTreeClassifier(random_state=0)
    # dt = dt.fit(fs_x, fs_y)
    # dfscores = pd.DataFrame(dt.feature_importances_)
    # dfcolumns = pd.DataFrame(fs_x.columns)
    # #concat two dataframes for better visualization 
    # featureScores = pd.concat([dfcolumns,dfscores],axis=1)
    # featureScores.columns = ['Feature','Importance']  #naming the dataframe columns
    # print("Decision Trees")
    # print(featureScores.nlargest(5,'Importance'))  #print 5 best features

    # sys.exit()

    # dt = time.process_time() - start

    # start = time.process_time()

    # # Extra Trees Classifier
    # forest = ExtraTreesClassifier(random_state=0)
    # forest = forest.fit(fs_x, fs_y)
    # dfscores = pd.DataFrame(forest.feature_importances_)
    # dfcolumns = pd.DataFrame(fs_x.columns)
    # #concat two dataframes for better visualization 
    # featureScores = pd.concat([dfcolumns,dfscores],axis=1)
    # featureScores.columns = ['Feature','Importance']  #naming the dataframe columns
    # print("Extra Trees")
    # print(featureScores.nlargest(5,'Importance'))  #print 5 best features

    # et = time.process_time() - start

    # start = time.process_time()

    # # Random Forest Classifier
    # forest = RandomForestClassifier(random_state=0)
    # forest = forest.fit(fs_x, fs_y)
    # dfscores = pd.DataFrame(forest.feature_importances_)
    # dfcolumns = pd.DataFrame(fs_x.columns)
    # #concat two dataframes for better visualization 
    # featureScores = pd.concat([dfcolumns,dfscores],axis=1)
    # featureScores.columns = ['Feature','Importance']  #naming the dataframe columns
    # print("Random Forest")
    # print(featureScores.nlargest(5,'Importance'))  #print 5 best features

    # rf = time.process_time() - start

    # start = time.process_time()

    # # Lasso
    # lasso = Lasso(random_state=0)
    # lasso = lasso.fit(fs_x, fs_y)
    # dfscores = pd.DataFrame(np.absolute(lasso.coef_))
    # dfcolumns = pd.DataFrame(fs_x.columns)
    # #concat two dataframes for better visualization 
    # featureScores = pd.concat([dfcolumns,dfscores],axis=1)
    # featureScores.columns = ['Feature','Importance']  #naming the dataframe columns
    # print("Lasso")
    # print(featureScores.nlargest(10,'Importance'))  #print 5 best features

    # # lasso = time.process_time() - start






    # start = time.process_time()
    # # # PCA
    # pca = PCA(n_components=1)
    # pca = pca.fit(fs_x, fs_y)
    # print(pca.components_.shape)


    # # dfscores = pd.DataFrame(pca.explained_variance_)
    # dfscores = pd.DataFrame(abs(pca.components_[0]))
    # dfcolumns = pd.DataFrame(fs_x.columns)
    # #concat two dataframes for better visualization 
    # featureScores = pd.concat([dfcolumns,dfscores],axis=1)
    # featureScores.columns = ['Feature','Variance']  #naming the dataframe columns
    # print("PCA")
    # print(featureScores.nlargest(5,'Variance'))  #print 5 best features
    # pca_time = time.process_time() - start


    # print(abs(pca.components_[0]))
    # print(np.abs(pca.components_[0]).argsort()[::-1][:5])

    # X = pca.fit_transform(fs_x)
    # df = pd.DataFrame(X)
    # df['Label'] = fs_y
    # # df.to_csv('test.csv', sep=",")
    # print(df)





    # estimator = linear_model.LogisticRegression(solver="liblinear", multi_class="ovr")
    # selector = GeneticSelectionCV(estimator,
    #                               cv=5,
    #                               verbose=1,
    #                               scoring="accuracy",
    #                               max_features=5,
    #                               n_population=50,
    #                               crossover_proba=0.5,
    #                               mutation_proba=0.2,
    #                               n_generations=40,
    #                               crossover_independent_proba=0.5,
    #                               mutation_independent_proba=0.05,
    #                               tournament_size=3,
    #                               n_gen_no_change=10,
    #                               caching=True,
    #                               n_jobs=-1)
    # selector = selector.fit(fs_x, fs_y)
    # print(selector.support_)

    # sys.exit()




    # # # print("Anova: ", anova)
    # # # print("LR Permutation Importance: ", lr_pi)
    # # # print("Decision Tree: ", dt)
    # # # print("Extra Trees: ", et)
    # # # print("Random Forest: ", rf)
    # # # print("Lasso: ", lasso)
    # print("PCA: ", pca_time)




    data = df.values
    X = data[:, :-1]
    X = X.astype(np.float64)

    nX = np.copy(X)
    rX = np.copy(X)

    feature_names = list(df.columns)
    del feature_names[-1]

    Y = data[:, -1]

    print("Calculating the mean values... ", end="", flush=True)
    features_mean = np.mean(rX, axis=0)
    print("Done!")

    print("Calculating the std values... ", end="", flush=True)
    features_std = np.std(rX, axis=0)
    print("Done!")

    features_std = np.where(features_std != 0, features_std, 1.0)

    print("Normalising the dataset... ", end="", flush=True)
    nX = (nX - features_mean) / features_std
    print("Done!")

    return nX, rX, Y, feature_names, technique
