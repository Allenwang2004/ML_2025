import pandas as pd
import sys
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def main():
    pred_csv = sys.argv[1]   # 模型預測的結果
    true_csv = sys.argv[2]   # 正確標籤（答案）

    # 讀取 CSV 檔案
    pred_df = pd.read_csv(pred_csv, header=None)  # 預測的 CSV 只有一欄
    true_df = pd.read_csv(true_csv, header=None)

    # 檢查資料長度
    if len(pred_df) != len(true_df):
        print("檔案長度不一致")
        return

    y_pred = pred_df[0].values
    y_true = true_df[0].values

    # 計算正確率
    acc = accuracy_score(y_true, y_pred)
    print(f"Accuracy: {acc:.4f}")

    # 額外：列印 classification report 和 confusion matrix
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, digits=4))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_true, y_pred))

if __name__ == "__main__":
    main()