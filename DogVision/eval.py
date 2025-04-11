import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

class ModelEvaluator:
    def __init__(self, model, test_generator, class_names):
        self.model = model
        self.test_generator = test_generator
        self.class_names = class_names
        
    def evaluate(self):
        self.probs = self.model.predict(self.test_generator)
        self.preds = np.argmax(self.probs, axis=1)
        self.labels = self.test_generator.classes
        
    def plot_confusion_matrix(self):
        cm = confusion_matrix(self.labels, self.preds)
        fig, ax = plt.subplots(figsize=(20, 20))
        ConfusionMatrixDisplay(cm, display_labels=self.class_names).plot(ax=ax)
        plt.xticks(rotation=90)
        return fig
        
    def get_class_accuracy(self):
        results = pd.DataFrame({
            "true": self.labels,
            "pred": self.preds
        })
        results["correct"] = results["true"] == results["pred"]
        return results.groupby("true")["correct"].mean().sort_values()  