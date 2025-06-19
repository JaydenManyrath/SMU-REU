import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

def display_samples_with_predictions(model, X_test, y_test, count=5, save_dir=None):
    """
    Displays sample images with plaintext predictions and optionally saves them.
    """
    for i in range(min(count, len(X_test))):
        img = X_test[i].reshape(32, 32)
        true_label = "Criminal" if y_test[i] == 1 else "General"

        # Plaintext prediction
        x_sample = X_test[i].reshape(1, -1).astype(np.float32)
        pred_plaintext = model.predict(x_sample)[0]
        pred_label = "Criminal" if pred_plaintext == 1 else "General"

        # Plot
        plt.imshow(img, cmap='gray')
        plt.title(f"True: {true_label} | Predicted: {pred_label}")
        plt.axis('off')
        if save_dir:
            save_path = os.path.join(save_dir, f"gui_sample_{i}_true_{true_label}_pred_{pred_label}.png")
            plt.savefig(save_path)
        plt.show()
