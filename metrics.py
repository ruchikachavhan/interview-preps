import numpy as np
import sklearn
from sklearn.metrics import confusion_matrix as cm


pred = [1, 0, 1, 2, 2, 6, 2, 3, 4, 1, 5, 4, 6, 5]
true = [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6]

#  Number of classes is 7 
# Therefore, confusion matrix will be 7 x 7
num_classes = 7
confusion_matrix = np.zeros((num_classes, num_classes))

for i in range(len(true)):
    confusion_matrix[true[i]][pred[i]] += 1

print(confusion_matrix)

# Compute class-wise accuracy
accuracy = np.einsum('ii->i', confusion_matrix)
total_samples = np.sum(confusion_matrix, axis = 1)
accuracy = accuracy/total_samples

# Compute class-wise precision

tps = np.einsum('ii->i', confusion_matrix)
fps, fns = [], []
for c in range(num_classes):
    fps.append(sum(confusion_matrix[:, c]) - confusion_matrix[c,c])
    fns.append(sum(confusion_matrix[c, :]) - confusion_matrix[c,c])

fps, fns = np.array(fps), np.array(fns)
precision = tps/(tps + fps)
recall = tps/(tps + fns)
print(precision)
print(recall)

f1_score = 2 * np.multiply(precision, recall)/(precision + recall)
print(f1_score)

# Metrics for semantic segmentation
y_pred = np.array([[1, 1, 1], [0, 0, 1], [0, 0, 1]])
y_true = np.array([[1, 0, 1], [0, 0, 1], [1, 0, 1]])


intersection = np.logical_and(y_true, y_pred).astype(np.int16)
union = np.logical_or(y_true, y_pred).astype(np.int16)
miou = np.sum(intersection)/np.sum(union)
print(miou)
