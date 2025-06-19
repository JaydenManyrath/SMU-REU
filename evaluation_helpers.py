import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc

def save_summary(text, filename):
    with open(filename, "w") as f:
        f.write(text)

def plot_confusion_matrix(y_true, y_pred, save_path):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["General", "Criminal"])
    disp.plot()
    plt.title("Confusion Matrix (Plaintext)")
    plt.savefig(save_path)
    plt.close()

def plot_accuracy_comparison(plain_acc, fhe_acc, save_path):
    plt.figure(figsize=(6, 4))
    plt.bar(["Plaintext", "FHE"], [plain_acc, fhe_acc], color=["green", "blue"])
    plt.ylabel("Accuracy (%)")
    plt.title("Plaintext vs FHE Accuracy")
    plt.ylim(0, 100)
    plt.savefig(save_path)
    plt.close()

def plot_roc_curve(y_true, y_score, save_path):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange',
             lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([-0.01, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.savefig(save_path)
    plt.close()
