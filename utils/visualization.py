import matplotlib.pyplot as plt

def plot_robustness(angles, acc_clean, acc_aug):
    plt.plot(angles, acc_clean, label="Clean Model")
    plt.plot(angles, acc_aug, label="Augmented Model")

    plt.xlabel("Rotation Angle")
    plt.ylabel("Accuracy")
    plt.title("Robustness Curve")
    plt.legend()
    plt.show()