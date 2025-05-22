import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

def evaluate_emotion_detection(log_path, gt_path, person_ids):
    log_df = pd.read_csv(log_path)
    gt_df = pd.read_csv(gt_path)

    y_true = []
    y_pred = []

    for _, row in gt_df.iterrows():
        frame = row["Frame_Number"]
        for pid in person_ids:
            gt_emotion = row[f"GT_ID_{pid}"]
            if gt_emotion == "Skipped":
                continue

            # Get detected emotion for the same frame and person ID
            detected = log_df[(log_df["frame_number"] == frame) & (log_df["id"] == pid)]
            if detected.empty:
                continue
            pred_emotion = detected.iloc[0]["emotion"]

            y_true.append(gt_emotion)
            y_pred.append(pred_emotion)

    # Compute metrics
    report = classification_report(y_true, y_pred, labels=list(set(y_true + y_pred)), zero_division=0)
    matrix = confusion_matrix(y_true, y_pred, labels=list(set(y_true + y_pred)))
    return report, matrix

person_ids1 = [1, 2, 3, 4, 5, 6]
person_ids2 = [1, 2, 3, 4]
report1, matrix1 = evaluate_emotion_detection("videos/classroom_log.csv", "videos/classroom_gt.csv", person_ids1)
report2, matrix2 = evaluate_emotion_detection("videos/classroom4_log.csv", "videos/classroom4_gt.csv", person_ids2)

print("Video 1 Performance:\n", report1)
print("Confusion Matrix for Video 1:\n", matrix1)
print("Video 2 Performance:\n", report2)
print("Confusion Matrix for Video 2:\n", matrix2)
