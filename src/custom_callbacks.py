from keras.callbacks import Callback
from sklearn.metrics import roc_auc_score


class IntervalEvaluation(Callback):  # pylint: disable=R0903
    """Computes ROC AUC metrics"""

    def __init__(self, validation_data=()):
        super(Callback, self).__init__()  # pylint: disable=E1003

        self.x_val, self.y_val = validation_data
        self.aucs = []

    def on_epoch_end(self, epoch, logs={}):
        """
        Count ROC AUC score at the end of each epoch
        """
        y_pred = None
        if hasattr(self.model, 'predict_proba'):
            # for Sequentional models
            y_pred = self.model.predict_proba(self.x_val, verbose=0)
        else:
            # for models that was created using functional API
            y_pred = self.model.predict(self.x_val, verbose=0)
        self.aucs.append(roc_auc_score(self.y_val, y_pred))
        print(
            '\repoch: {:d} - ROC AUC: {:.6f}'.format(epoch + 1, self.aucs[-1]))
