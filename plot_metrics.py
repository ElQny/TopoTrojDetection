import os

import matplotlib.pyplot as plt
import pandas as pd

def boxplot(df, paramname):
    fig, ax = plt.subplots(3, sharex=True)

    df.boxplot(column='acc', by='param_value', ax=ax[0], grid=True)
    df.boxplot(column='auc', by='param_value', ax=ax[1], grid=True)
    df.boxplot(column='ce', by='param_value', ax=ax[2], grid=True)

    ax[0].set_ylabel('ACC (%)')
    ax[1].set_ylabel('AUC (%)')
    ax[2].set_ylabel('CE')

    for a in ax:
        a.set_title('')
        a.set_xlabel('')
        a.grid(True, axis='y', alpha=0.25)

    ax[2].set_xlabel('Parameterwert')

    plt.suptitle(f'Einfluss von {paramname} auf ACC, AUC und CE')
    plt.tight_layout(h_pad=1.2)
    plt.show()


def lineplot(df_joint, paramname):
    fig, ax = plt.subplots(3, sharex=True)
    fig.suptitle(f'Einfluss von {paramname} auf ACC, AUC und CE')



    xtitle = 'Parameterwert'
    df_len = df_joint.shape[0]

    df_joint.plot(
        kind='line',
        ax=ax[0],
        x='param_value',
        y='acc_mean',
        yerr='acc_std',
        legend=False,
        ylabel='ACC (%)',
        xlabel=xtitle,
        xticks=range(df_len)
    )
    df_joint.plot(
        kind='line',
        ax=ax[1],
        x='param_value',
        y='auc_mean',
        yerr='auc_std',
        legend=False,
        ylabel='AUC (%)',
        xlabel=xtitle,
        xticks=range(df_len)
    )
    df_joint.plot(
        kind='line',
        ax=ax[2],
        x='param_value',
        y='ce_mean',
        yerr='ce_std',
        legend=False,
        ylabel='CE',
        xlabel=xtitle,
        xticks=range(df_len)
    )

    for a in ax:
        a.grid(True, axis='y', alpha=0.25)

    plt.tight_layout(h_pad=1.2)
    plt.show()

def plot_csv(csvfile, paramname):
    if not csvfile.endswith('.csv'):
        raise Exception(f'File {csvfile} is invalid')
    if not os.path.exists(csvfile):
        raise Exception(f'File {csvfile} does not exist')
    df:pd.DataFrame = pd.read_csv(csvfile)
    df_grouped = df.groupby('param_value')
    df_mean = df_grouped[['acc', 'auc', 'ce']].mean()
    df_std = df_grouped[['acc', 'auc', 'ce']].std()

    df_mean = df_mean.rename(columns= {
        'acc' : 'acc_mean', 'auc':'auc_mean', 'ce': 'ce_mean'
    })
    df_std = df_std.rename(columns = {
        'acc' : 'acc_std', 'auc' : 'auc_std', 'ce' : 'ce_std'
    })

    df_joint = df_mean.join(df_std, how='inner', on='param_value').reset_index()

    # boxplot(df,paramname)
    lineplot(df_joint, paramname) #uncomment to use

def main():
    try:
        filename = 'corr_metric_old2.csv'
        parametername = 'Korrelationsmetrik'
        filepath = os.path.join(os.getcwd(), 'tmp', filename)
        plot_csv(filepath, parametername)
    except Exception as e:
        print(f'Exception occured: {e}')

if __name__ == '__main__':
    main()