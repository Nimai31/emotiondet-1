import os
import pandas as pd
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas


def generate_pdf(session_id, df, saved_faces):
    from reportlab.lib.pagesizes import letter
    from reportlab.lib import colors
    from reportlab.lib.utils import ImageReader
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
    from reportlab.lib.styles import getSampleStyleSheet
    import matplotlib.pyplot as plt
    import io

    pdf_path = f"emotion_analytics_{session_id}.pdf"
    doc = SimpleDocTemplate(pdf_path, pagesize=letter)
    elements = []
    styles = getSampleStyleSheet()

    students = df['id'].unique()

    for student_id in students:
        student_df = df[df['id'] == student_id]

        # Light gray background for each student section
        elements.append(Spacer(1, 10))
        bg_table = Table([['']], colWidths=[doc.width], rowHeights=20)
        bg_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), colors.lightgrey)
        ]))
        elements.append(bg_table)

        # Student Header
        header = Paragraph(f"<b>Student ID: {student_id}</b>", styles['Title'])
        elements.append(header)
        elements.append(Spacer(1, 10))

        # Face Image
        face_path = saved_faces.get(student_id)
        if face_path and os.path.exists(face_path):
            face_img = Image(face_path, width=100, height=100)
            elements.append(face_img)
            elements.append(Spacer(1, 10))

        # Emotion counts
        emotion_counts = student_df['emotion'].value_counts()
        total = emotion_counts.sum()

        # Generate Pie Chart
        pie_buf = io.BytesIO()
        fig1, ax1 = plt.subplots(figsize=(4, 4))
        ax1.pie(emotion_counts, labels=emotion_counts.index, autopct='%1.1f%%', startangle=90)
        ax1.set_title("Time Spent in Each Emotion")
        plt.tight_layout()
        plt.savefig(pie_buf, format='png')
        plt.close()
        pie_buf.seek(0)

        # Generate Bar Chart - Duration
        bar_buf = io.BytesIO()
        fig2, ax2 = plt.subplots(figsize=(4, 4))
        emotion_counts.plot(kind='bar', ax=ax2, color='lightcoral')
        ax2.set_xlabel("Emotion")
        ax2.set_ylabel("Frames Detected")
        ax2.set_title("Emotion Duration")
        plt.tight_layout()
        plt.savefig(bar_buf, format='png')
        plt.close()
        bar_buf.seek(0)

        # Timeline Scatter Plot
        timeline_buf = io.BytesIO()
        fig3, ax3 = plt.subplots(figsize=(6, 2))
        student_df.loc[:,'timestamp'] = pd.to_datetime(student_df['timestamp'])
        ax3.scatter(student_df['timestamp'], student_df['emotion'].astype(str), c='green')
        ax3.set_title('Emotion Over Time')
        ax3.set_xlabel('Time')
        ax3.set_ylabel('Emotion')
        plt.xticks(rotation=30, ha='right')
        plt.tight_layout()
        plt.savefig(timeline_buf, format='png')
        plt.close()
        timeline_buf.seek(0)

        # Average Confidence Bar Chart
        avg_conf = student_df.groupby('emotion')['confidence'].mean()
        conf_buf = io.BytesIO()
        fig4, ax4 = plt.subplots(figsize=(4, 4))
        avg_conf.plot(kind='bar', color='orange')
        ax4.set_ylabel('Average Confidence (%)')
        ax4.set_title('Average Confidence per Emotion')
        plt.tight_layout()
        plt.savefig(conf_buf, format='png')
        plt.close()
        conf_buf.seek(0)

        # Add Charts
        elements.append(Image(pie_buf, width=200, height=200))
        elements.append(Image(bar_buf, width=200, height=200))
        elements.append(Spacer(1, 10))
        elements.append(Image(timeline_buf, width=400, height=100))
        elements.append(Spacer(1, 10))
        elements.append(Image(conf_buf, width=200, height=200))
        elements.append(Spacer(1, 20))

        # Insights
        bored_time = emotion_counts.get('bored', 0)
        confused_time = emotion_counts.get('confused', 0)
        frustrated_time = emotion_counts.get('frustrated', 0)
        happy_time = emotion_counts.get('happy', 0)
        focused_time = emotion_counts.get('focused', 0)

        insights = ""
        if bored_time / total > 0.25:
            insights += "📌 The student might be <b>bored</b>.\n"
        if (confused_time + frustrated_time) / total > 0.25:
            insights += "❓ The student might have <b>doubts</b>.\n"
        if (happy_time + focused_time) / total > 0.3:
            insights += "✅ The student appears <b>engaged and understanding</b> the material.\n"

        if insights:
            elements.append(Paragraph(insights, styles['Normal']))
            elements.append(Spacer(1, 10))

        # Top 3 Emotions
        top3 = emotion_counts.head(3)
        top3_text = "<b>🏆 Top 3 Emotions:</b><br/>"
        for i, (emo, count) in enumerate(top3.items(), 1):
            top3_text += f"{i}. {emo.capitalize()} - {count} times<br/>"
        elements.append(Paragraph(top3_text, styles['Normal']))
        elements.append(Spacer(1, 20))

    # Build PDF
    doc.build(elements)

    return pdf_path
