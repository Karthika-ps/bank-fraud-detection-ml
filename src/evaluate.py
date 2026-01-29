from sklearn.metrics import classification_report, confusion_matrix

def evaluate_model(model, X_val, y_val):
    preds = model.predict(X_val)
    report = classification_report(y_val, preds, output_dict=False)
    matrix = confusion_matrix(y_val, preds)

    print(report)
    print("Confusion Matrix:")
    print(matrix)

    return report, matrix
