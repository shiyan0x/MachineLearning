from sklearn.metrics import confusion_matrix

#True answer
y_true = [1,0,1,0,1,1]

#model_predict
y_pred = [1,0,1,1,1,0]

#evaluation
cm = confusion_matrix(y_true, y_pred)
print("True|Predicted")
print("   1  0")
print(cm)

#labeling

TP, FP, FN, TN = cm.ravel()

print("True Positive ", TP)
print("False Positive ", FP)
print("False Negative ", FN)
print("True Negative ", TN)
