
def errorbar(values, acc_mean, acc_std, auc_mean, auc_std, ce_mean, ce_std, label):
    fig, ax = plt.subplots(1, 2, figsize=(12,6))

    #ACC/AUC
    ax[0].errorbar(values, acc_mean, yerr=acc_std, marker='o', label='ACC')
    ax[0].errorbar(values, auc_mean, yerr=auc_std, marker='s', label='AUC')
    ax[0].set_xlabel(label)
    ax[0].set_ylabel('Accuracy und AUC in %')
    ax[0].set_title(f'ACC und AUC bei Variation von {label}')
    ax[0].legend()
    ax[0].grid(True)

    #CE
    ax[1].errorbar(values, ce_mean, yerr=ce_std, marker='o', label='CE')
    ax[1].set_xlabel(label)
    ax[1].set_ylabel('Cross-Entropy')
    ax[1].set_title(f'CE bei Variation von {label}')
    ax[1].legend()
    ax[1].grid(True)

    plt.tight_layout()
    plt.show()