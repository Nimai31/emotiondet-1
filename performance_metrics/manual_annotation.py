import pandas as pd
import os

# --- Config ---
output_csv_path = "manual_annotations_progress.csv"
emotions_list = ['bored', 'confused', 'focused', 'frustrated', 'happy', 'neutral', 'surprised']
person_ids_to_annotate = [1, 2, 3, 4, 5, 6]

# --- Load or Initialize ---
def load_annotations(path):
    if os.path.exists(path):
        try:
            df = pd.read_csv(path)
            df["Frame_Number"] = df["Frame_Number"].astype(int)
            return df
        except Exception as e:
            print(f"Error loading CSV: {e}")
    return pd.DataFrame(columns=["Frame_Number"] + [f"GT_ID_{pid}" for pid in person_ids_to_annotate])

def save_annotations(df, path):
    df.to_csv(path, index=False)
    print(f"Saved to {path}")

# --- Main Annotation Loop ---
def annotate():
    df = load_annotations(output_csv_path)

    while True:
        try:
            user_input = input("\nEnter frame number to annotate (or 'q' to quit): ").strip().lower()
            if user_input == 'q':
                break
            frame_num = int(user_input)

            if frame_num in df["Frame_Number"].values:
                print(f"Frame {frame_num} already annotated. Overwriting.")

            frame_data = {"Frame_Number": frame_num}
            for pid in person_ids_to_annotate:
                print(f"\nPerson ID {pid}:")
                for i, emo in enumerate(emotions_list):
                    print(f"  {i}: {emo}")
                print("  s: Skip")

                while True:
                    choice = input(f"Select emotion (0-{len(emotions_list)-1}) for ID {pid}, or 's' to skip: ").strip().lower()
                    if choice == 's':
                        frame_data[f"GT_ID_{pid}"] = "Skipped"
                        break
                    try:
                        idx = int(choice)
                        if 0 <= idx < len(emotions_list):
                            frame_data[f"GT_ID_{pid}"] = emotions_list[idx]
                            break
                    except:
                        pass
                    print("Invalid input.")

            # Save this frame's annotations
            df = pd.concat([df[df["Frame_Number"] != frame_num], pd.DataFrame([frame_data])])
            save_annotations(df, output_csv_path)

        except ValueError:
            print("Invalid frame number. Try again.")

    print("Annotation session ended.")

if __name__ == "__main__":
    annotate()
