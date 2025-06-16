import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
from PIL import Image
import gc
import io
import os
import pickle
from xgboost import XGBClassifier, DMatrix
from textwrap import wrap
seed = 42

def get_top_shap_feature(clf, df, feature_names, ntop):
    """
    This function will return the top n features with highest shaply values
    clf: model object
    df: pandas dataframe, input features only
    feature_names: list, feature names of input df
    ntop: number of top features to return
    """
    gc.collect()
    explainer = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(df)
    if isinstance(shap_values, list):
        shap_df = pd.DataFrame([np.mean(np.abs(sv), axis=0) for sv in shap_values], columns=feature_names)
        shap_mean_df = shap_df.mean().sort_values(ascending=False)
    else:
        shap_df = pd.DataFrame(np.abs(shap_values), columns=feature_names)
        shap_mean_df = shap_df.mean().sort_values(ascending=False)

    top_features = shap_mean_df.head(ntop).index.tolist()

    gc.collect()
    return top_features

def plot_shap_var_imp(clf, df, descriptive_name=True, max_display=None, path_name=None):
    '''
    Function to create shapley value feature importance plot
    clf: model object
    df: pandas dataframe, input features only
    max_display = int, change the number of top predictive features to display
    path_name = full path to save the file, including the file name e.g. /docs/shap.jpeg
                use jpeg format for better compression
    '''
    explainer = shap.TreeExplainer(clf)

    column_name = df.columns
    shap_df = pd.DataFrame(explainer.shap_values(df),columns=column_name)

    if max_display:
        max_display = min(max_display,len(shap_df.columns))
    else:
        max_display = len(shap_df.columns)

    shap_mean_df = shap_df.abs().mean(axis=0).sort_values(ascending=False)[:max_display]

    num_zero_shap = (shap_mean_df == 0).sum()

    fig, ax = plt.subplots()
    plt.subplots_adjust(top=0.8, bottom=0.2, left=0.1)
    plt.gcf().set_size_inches(8, min(0.5*len(shap_mean_df),655))
    ax.barh(np.arange(len(shap_mean_df)), shap_mean_df, align='center')
    ax.set_yticks(np.arange(len(shap_mean_df)), labels=shap_mean_df.index)
    plt.ylim([-1,np.arange(len(shap_mean_df)).size])
    ax.invert_yaxis()
    if path_name:
        plt.savefig(path_name,dpi=75,bbox_inches='tight')
    print(f"{num_zero_shap} variables in this plot have SHAP value = 0.")
    return None


def _calc_pd_bins(arr):

    unique_vals = np.unique(arr)
    neg_unique_vals = unique_vals[unique_vals<=0]
    pos_unique_vals = unique_vals[unique_vals>=0]

    if len(pos_unique_vals) > 10:
        bins = np.unique(np.percentile(arr[arr>=0], np.arange(0, 100, 5)))
        bins_round = np.array([bin if 0 < bin < 1 else np.round(bin) for bin in bins])
    else:
        bins = pos_unique_vals
        bins_round = np.array([bin if 0 < bin < 1 else np.round(bin) for bin in bins])

    if len(neg_unique_vals) <= 20:
        bins_round = np.append(neg_unique_vals, bins_round)
    else:
        bins = np.unique(np.percentile(arr, np.arange(0, 100, 5)))
        bins_round = np.array([bin if 0 < bin < 1 else np.round(bin) for bin in bins])

    bins_round = np.append(bins_round, np.array(np.max(arr)))

    return np.unique(bins_round)


def _calc_pd_values(model, df, feature, bins, weight=None):

    df = df.copy()

    pd_df = pd.DataFrame(columns=[feature, 'pd'])

    for j in bins:

        df[feature] = j
        try:
            pd_i = model.predict_proba(df)[:,1]
        except:
            pd_i = model.predict(df)
        pd_j = np.sum(pd_i*weight)/np.sum(weight)
        pd_df = pd.concat([pd_df, pd.DataFrame({feature:j, 'pd': pd_j}, index=[0])], ignore_index=True)
        pd_df['pd_diff'] = pd_df['pd'] - pd_df.iloc[0]['pd'].min()

    pd_df[feature] = [np.round(row,3) if 0 < row < 1 else int(row) for row in pd_df[feature]]
    pd_df[feature] = pd_df[feature].astype(str)

    del df
    gc.collect()

    return pd_df

def plot_wt_pdp(model, df,feature, index="", weight=None, path=None, clusters=1, sample_frac=0.20):
    """
    !!! Make sure df has unique indices, program uses indices for creating clusters !!!
    The function will create a partial-dependence plot for the feature based on the classifier and input data
    model:  model with scikit-learn API, with either predict_proba or predict method
    df: pd.DataFrame, data used for calculating predictions, make sure only variables used to fit the model are in the data
    feature: feature whose pdp needs to be plotted
    index: prefix index of the file name
    num_bins: int, number of bins of the feature to be used for calculating partial dependence
    weight: pd.Series, sampling weight of the observations
    path: by default the function only displays the plot, if path is provided, the plots will be saved to the dir provided
    clusters: int, number of cluster partial dependence lines to plot
    sample_frac: fraction of sample to be used for calculating partial dependence
    """

    feat=feature.split(":")[0]

    bins = _calc_pd_bins(df[feat])

    df = df.copy()

    try:
        df['preds'] = model.predict_proba(df)[:,1]
    except:
        df['preds'] = model.predict(df)

    if weight is None:
        df['weight'] = pd.Series(np.ones(df.shape[0]), df.index)
    else:
        df['weight'] = weight

    df = df.sample(frac=sample_frac, weights=weight, random_state=seed)
    df = df.sort_values(by='preds')

    weight = df['weight']
    df = df.drop(labels=['preds','weight'], axis='columns')

    pd_df = _calc_pd_values(model, df, feat, bins, weight)

    cum_weight = np.cumsum(weight) / np.sum(weight)
    idx = np.searchsorted(cum_weight, np.linspace(0, 1, clusters, endpoint=False)[1:])
    chunks = np.split(df, idx)

    if clusters > 1:
        for i, chunk in enumerate(chunks):
            pd_chunks = _calc_pd_values(model, chunk, feat, bins, weight.loc[chunk.index])
            pd_df = pd_df.merge(pd_chunks, how='inner', on=feat, suffixes=['','_'+str(i)])

    plt.figure(figsize=(12, 7))
    plt.title("\n".join(wrap(feature, 80)), fontdict={'fontsize':12})
    plt.plot(pd_df[feat], pd_df['pd_diff'], color='yellow', linewidth=5)
    plt.plot(pd_df[feat], pd_df['pd_diff'], color='black', linewidth=3)
    plt.xticks(ticks=pd_df[feat], rotation=90)
    plt.grid(visible=True, axis='both', color='dodgerblue', linestyle='dotted', linewidth=1)

    if clusters > 1:
        chunk_pd_diff_cols = pd_df.columns[pd_df.columns.str.startswith('pd_diff_')]
        colors = iter(['gold','blue','green','red','wheat','blueviolet','deeppink','gray','brown','rebeccapurple'])
        for col in chunk_pd_diff_cols:
            plt.plot(pd_df[feat], pd_df[col], color=next(colors), linestyle='dashed', linewidth=3)

    if index != "":
        index = str(index) + "_"

    if sys.version_info[1] >= 10:
        left = 80
        top = 40
        right = 1100
        bottom = 700
    else:
        left = 50
        top = 20
        right = 800
        bottom = 490

    if os.name != 'posix':
        feature = feature.replace(":","_")

    if path is not None:
        plt.savefig(path + index + feature + ".png")
        im = Image.open(path + index + feature + ".png")
        cropped = im.crop((left, top, right, bottom))
        cropped.save(path + index + feature + ".png")

    gc.collect()

    return None

#######################################################################################3

data_dir = '/home/cwang/Documents/github/Data/ir7/'
model_dir = '/home/cwang/Documents/github/decision-sciences-ir7/Models/'

df_final_train=pd.read_pickle(data_dir+'ir7_final_train_ds.dat')
df_final_train=df_final_train.reset_index(drop=True)
col_in_model = pickle.load(open(file='/home/cwang/Documents/github/decision-sciences-ir7/Columns/ir7_kgb_cols_r2.pkl', mode='rb'))

X_train = df_final_train[col_in_model]
weight = df_final_train['weight']

sk_clf_1 = XGBClassifier()
sk_clf_1.load_model(model_dir + 'ir7_final_v1.bin')

plot_shap_var_imp(sk_clf_1,X_train,max_display=100)
plot_wt_pdp(sk_clf_1, X_train, 'br20s', weight=weight, clusters =5,sample_frac=0.1)
