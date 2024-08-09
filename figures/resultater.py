import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np
import seaborn as sns


def barplot_aggragated(dfs, score):
    # plot styling
    barWidth = 0.3
    color = '#25BE85'
    fontsize = 15
    error_kw = dict(lw=5, capsize=5, capthick=3)
    score_names_formatted = {'dice_score': 'Dice Score',
                             'haus_score': 'Hausdorff Distance'}

    names = [name for name in dfs.keys()]
    means = [df.mean()[score] for df in dfs.values()]
    stds = [df.std()[score] for df in dfs.values()]
    Ns = [len(df.index) for df in dfs.values()]

    higher = np.array([stats.norm.interval(0.95, loc=means[i], scale=stds[i] / np.sqrt(Ns[i]))[1] for i in range(len(Ns))])
    lower = np.array([stats.norm.interval(0.95, loc=means[i], scale=stds[i] / np.sqrt(Ns[i]))[0] for i in range(len(Ns))])
    yerr = higher-lower

    fig = plt.figure(figsize=(8, 10))
    plt.bar(names, means, yerr=yerr, width=barWidth, color=color, error_kw=error_kw)
    plt.title(score_names_formatted[score], fontsize=fontsize)
    plt.ylabel(score_names_formatted[score], fontsize=fontsize-3)
    plt.xticks(fontsize=fontsize)
    plt.tight_layout()
    plt.show()


def barplot_grouped(dfs, score):
    # plot styling
    barWidth = 0.3
    # color = '#25BE85'
    colors = ['#3274a1', '#e1812c', '#3a923a']
    fontsize = 25
    legend_size = 26
    error_kw = dict(lw=3, capsize=5, capthick=2)
    formatted = {'dice_score': 'Dice Score',
                             'haus_score': 'Hausdorff Distance'}


    fig = plt.figure(figsize=(10, 6))
    for j, (model_name, df) in enumerate(dfs.items()):
        group = df.groupby("scan_name")[score]
        scan_names = group.mean().keys().values
        means = group.mean().values
        stds = group.std().values
        Ns = group.count().values

        X_axis = np.arange(len(scan_names)+1)

        a = 2
        higher = np.array(
            [stats.norm.interval(0.95, loc=means[i], scale=stds[i] / np.sqrt(Ns[i]))[1] for i in range(len(Ns))])
        lower = np.array(
            [stats.norm.interval(0.95, loc=means[i], scale=stds[i] / np.sqrt(Ns[i]))[0] for i in range(len(Ns))])
        yerr = higher - lower


        plt.bar(X_axis+j*barWidth, list(means)+[0], yerr=list(yerr)+[0], width=barWidth, error_kw=error_kw, label=model_name, color=colors[j])

    plt.xticks(X_axis+0.5*barWidth*(len(dfs.keys())-1), list(scan_names)+[""], fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.title("{} grouped on different scans".format(formatted[score]), fontsize=fontsize+4)
    plt.ylabel(formatted[score], fontsize=fontsize)
    plt.legend(fontsize=legend_size)
    # plt.tight_layout(w_pad=100)
    plt.show()

        # means = group.mean

def violinplot_aggragated(dfs, score):
    barWidth = 0.3
    # color = '#25BE85'
    fontsize = 21
    formatted = {'dice_score': 'Dice Score',
                             'haus_score': 'Hausdorff Distance'}

    pen = sns.load_dataset("penguins")
    # sns.violinplot(data=pen, x="body_mass_g", y="sex")
    #     # plt.show()

    plot_df = {'scan_name': [], formatted['dice_score']: [],
               formatted['haus_score']: [], 'model': []}

    for j, (model_name, df) in enumerate(dfs.items()):
        plot_df['scan_name'].extend(list(df['scan_name'].values))
        plot_df[formatted['dice_score']].extend(list(df['dice_score'].values))
        plot_df[formatted['haus_score']].extend(list(df['haus_score'].values))
        plot_df['model'].extend([model_name]*len(list(df['scan_name'].values)))

    plot_df = pd.DataFrame(data=plot_df)

    sns.violinplot(data=plot_df, x=formatted[score], y="model")
    plt.xlabel(formatted[score], fontsize=fontsize)
    plt.ylabel('', fontsize=0.1)
    # plt.ylabel('off')
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize+4)
    plt.title('Violin plot for {}'.format(formatted[score]), fontsize=fontsize+4)
    plt.show()





if __name__ == "__main__":


    dfs = {"regular": pd.read_csv(r"../resultater_2d.csv"),
           'interpolation': pd.read_csv(r"../resultater_3d.csv"),
           'contextual': pd.read_csv(r"../resultater_contextual.csv")}
    # barplot_aggragated(dfs, 'dice_score')
    barplot_grouped(dfs, 'dice_score')
    # violinplot_aggragated(dfs, 'dice_score')