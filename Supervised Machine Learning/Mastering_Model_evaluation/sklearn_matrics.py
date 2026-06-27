from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

#True answer
y_true = [1,0,1,0,1,1]

#model_predict
y_pred = [1,0,1,1,1,0]

#evaluation
print("Accuracy", accuracy_score(y_true, y_pred))
print("precision", precision_score(y_true, y_pred))
print("recall", recall_score(y_true, y_pred))
print("f1_score", f1_score(y_true, y_pred))
