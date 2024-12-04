"""
This module implements a Flask web application for sentiment analysis.
It allows users to submit text, which is then split into sentences and stored in MongoDB.
Results can later be fetched if the analysis is complete.
"""

import os
import uuid
from datetime import datetime
from flask import Flask, render_template, request, jsonify, url_for
from pymongo import MongoClient, errors
import nltk
from nltk.tokenize import sent_tokenize
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib
import io
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib.utils import ImageReader
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
import base64
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.application import MIMEApplication
from email.mime.text import MIMEText
from wordcloud import WordCloud

app = Flask(__name__)

# Download NLTK data for sentence tokenization
nltk.download("punkt")

# Connect to MongoDB
mongo_uri = os.getenv("MONGO_URI")
client = MongoClient(mongo_uri)  # Adjust the connection string if necessary
db = client["sentiment"]  # Database name
collection = db["texts"]  # Collection name
matplotlib.use('Agg')

@app.route("/")
def index():
    """Render the index page."""
    return render_template("index.html")


@app.route("/checkSentiment", methods=["POST"])
def submit_sentence():
    """
    Process the input sentence, split into individual sentences,
    and store them in MongoDB with a unique request_id.
    """
    data = request.get_json()
    paragraph = data.get("sentence", "").strip()

    if not paragraph:
        return jsonify({"error": "Input text is empty."}), 400

    # Split paragraph into individual sentences
    sentences = sent_tokenize(paragraph)

    # Generate a unique request_id
    request_id = str(uuid.uuid4())

    # Create sentence entries with status "pending" and analysis as null
    sentence_entries = [
        {"sentence": sentence, "status": "pending", "analysis": None}
        for sentence in sentences
    ]

    # Create the document structure
    document = {
        "request_id": request_id,
        "sentences": sentence_entries,
        "overall_status": "pending",
        "timestamp": datetime.now(),
    }

    try:
        # Insert into MongoDB
        result = collection.insert_one(document)
        print("Inserted Document ID:", result.inserted_id)
    except errors.PyMongoError as e:
        print(f"Error inserting document: {e}")
        return jsonify({"error": "Database insertion error."}), 500

    # Return the request_id for fetching results later
    return jsonify({"request_id": request_id})



@app.route("/get_analysis", methods=["GET"])
def get_analysis():
    """
    Fetch the sentiment analysis result for a given request_id.
    Returns both processed and error documents.
    """
    request_id = request.args.get("request_id")
    print(f"Received request to get analysis for request_id: {request_id}")

    document = collection.find_one({"request_id": request_id})
    if document:
        print("Document found:", document)
        document["_id"] = str(document["_id"])
        if document.get("overall_status") == "processed":
            return jsonify(document)
        elif document.get("overall_status") == "error":
            return jsonify({"error": document.get("error_message", "Processing error.")}), 400
        else:
            print("Document not yet processed.")
            return jsonify({"message": "Analysis not yet complete."}), 202
    print("No analysis found for request_id:", request_id)
    return jsonify({"message": "No analysis found"}), 404


@app.route("/emotion_intensity/<string:request_id>", methods=["GET"])
def get_emtion_intensity(request_id):
    """
    Calculate the intensity of each emotion for the given document and return the result as JSON.
    """
    document = collection.find_one({"request_id": request_id})

    if not document:
        return jsonify({"error": "Document not found"}), 404
    
    # caculate emotion intensity
    emotion_intensity = defaultdict(float)
    for sentence in document.get("sentences", []):
        emotions = sentence.get("emotions", [])
        for emotion in emotions:
            emotion_intensity[emotion] += 1
    
    # normalize the values by the total number of sentences to get average intensity
    total_sentences = len(document.get("sentences", []))
    if total_sentences > 0:
        emotion_intensity = {k: v / total_sentences for k, v in emotion_intensity.items()}

    return jsonify(emotion_intensity)

# ----- PDF Generation Functions -----
def generate_plots(document):
    """
    Generate all the plots using matplotlib and return them as images.
    """
    images = {}

    # Sentiment Trend Line Plot
    try:
        sentiment_trend = document.get('sentiment_trend', [])
        if sentiment_trend:
            x = [point['sentence_index'] for point in sentiment_trend]
            y = [point['compound'] for point in sentiment_trend]

            plt.figure(figsize=(8, 4))
            plt.plot(x, y, marker='o', linestyle='-')
            plt.title('Sentiment Trend')
            plt.xlabel('Sentence Index')
            plt.ylabel('Compound Score')
            plt.grid(True)

            # Save plot to a BytesIO object
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            images['sentiment_trend'] = buf
            plt.close()
    except Exception as e:
        print(f"Error generating Sentiment Trend plot: {e}")

    # Sentiment Distribution Histogram
    try:
        sentences = document.get('sentences', [])
        compound_scores = [s['analysis']['compound'] for s in sentences if s.get('analysis')]
        if compound_scores:
            plt.figure(figsize=(8, 4))
            plt.hist(compound_scores, bins=20, color='blue', edgecolor='black')
            plt.title('Sentiment Distribution')
            plt.xlabel('Compound Score')
            plt.ylabel('Frequency')
            plt.grid(True)

            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            images['sentiment_distribution'] = buf
            plt.close()
    except Exception as e:
        print(f"Error generating Sentiment Distribution plot: {e}")

    # Word Cloud for Topics
    try:
        topics = document.get('topics', [])
        if topics:
            # Extract words and weights
            word_freq = {}
            for topic in topics:
                # topic[1] is the topic string
                words_weights = topic[1].split('+')
                for item in words_weights:
                    weight, word = item.strip().split('*')
                    word = word.replace('"', '').strip()
                    weight = float(weight)
                    word_freq[word] = word_freq.get(word, 0) + weight

            # Generate word cloud
            wc = WordCloud(width=800, height=400, background_color='white')
            wc.generate_from_frequencies(word_freq)

            plt.figure(figsize=(8, 4))
            plt.imshow(wc, interpolation='bilinear')
            plt.axis('off')

            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            images['topics_wordcloud'] = buf
            plt.close()
    except Exception as e:
        print(f"Error generating Topics Word Cloud: {e}")

    # Overall Emotions Pie Chart
    try:
        sentences = document.get('sentences', [])
        emotion_counts = defaultdict(int)
        for sentence in sentences:
            emotions = sentence.get('emotions', [])
            for emotion in emotions:
                emotion_counts[emotion] += 1

        if emotion_counts:
            labels = list(emotion_counts.keys())
            sizes = [emotion_counts[label] for label in labels]

            plt.figure(figsize=(6, 6))
            plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140)
            plt.title('Overall Emotional Distribution')

            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            images['overall_emotions'] = buf
            plt.close()
    except Exception as e:
        print(f"Error generating Overall Emotions Pie Chart: {e}")

    # Sentiment Intensity Bar Chart
    try:
        sentences = document.get('sentences', [])
        sentence_indices = list(range(1, len(sentences) + 1))
        compound_scores = [s['analysis']['compound'] for s in sentences if s.get('analysis')]

        if compound_scores:
            plt.figure(figsize=(8, 4))
            plt.bar(sentence_indices, compound_scores, color='skyblue')
            plt.title('Sentiment Intensity per Sentence')
            plt.xlabel('Sentence Index')
            plt.ylabel('Compound Score')
            plt.grid(True)

            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            images['sentiment_intensity'] = buf
            plt.close()
    except Exception as e:
        print(f"Error generating Sentiment Intensity Bar Chart: {e}")

    # Emotional Shifts Area Chart
    try:
        sentences = document.get('sentences', [])
        emotion_types = ["Happy", "Angry", "Surprise", "Sad", "Fear", "Neutral"]
        emotion_data = {emotion: [] for emotion in emotion_types}
        sentence_indices = list(range(1, len(sentences) + 1))

        for sentence in sentences:
            emotions = sentence.get('emotions', [])
            for emotion in emotion_types:
                emotion_data[emotion].append(1 if emotion in emotions else 0)

        if sentences:
            plt.figure(figsize=(8, 4))
            plt.stackplot(sentence_indices, *[emotion_data[emotion] for emotion in emotion_types], labels=emotion_types)
            plt.title('Emotional Shifts Over Sentences')
            plt.xlabel('Sentence Index')
            plt.ylabel('Emotion Intensity')
            plt.legend(loc='upper left')
            plt.grid(True)

            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            images['emotional_shifts'] = buf
            plt.close()
    except Exception as e:
        print(f"Error generating Emotional Shifts Area Chart: {e}")

    return images

def create_pdf(document, images):
    """
    Create a PDF report using ReportLab and return it as a BytesIO object.
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []

    title = Paragraph("Sentiment Analysis Report", styles['Title'])
    story.append(title)
    story.append(Spacer(1, 12))
    timestamp = document.get('timestamp', datetime.now()).strftime('%B %d, %Y at %H:%M:%S')
    timestamp_paragraph = Paragraph(f"Generated on: {timestamp}", styles['Normal'])
    story.append(timestamp_paragraph)
    story.append(Spacer(1, 12))
    summary = document.get('summary', 'No summary available.')
    summary_paragraph = Paragraph(f"<b>Executive Summary:</b> {summary}", styles['Normal'])
    story.append(summary_paragraph)
    story.append(Spacer(1, 12))

    # Add Plots
    for key, img_buf in images.items():
        # Add a heading for each plot
        heading = Paragraph(f"<b>{key.replace('_', ' ').title()}</b>", styles['Heading2'])
        story.append(heading)
        story.append(Spacer(1, 12))

        # Add the image
        img = Image(img_buf, width=6*inch, height=3*inch)
        story.append(img)
        story.append(Spacer(1, 24))

    # Add Named Entities Table
    try:
        sentences = document.get('sentences', [])
        data_for_table = []

        # Define a style for the table cells
        cell_style = styles['Normal']
        cell_style.fontSize = 10  # Adjust font size as needed
        cell_style.leading = 12   # Adjust line spacing as needed

        for idx, sentence_entry in enumerate(sentences):
            sentence_text = sentence_entry.get('sentence', '')
            entities = sentence_entry.get('entities', [])
            if entities:
                for entity in entities:
                    data_for_table.append([
                        Paragraph(f"Sentence {idx + 1}: {sentence_text}", cell_style),
                        Paragraph(entity.get('text', ''), cell_style),
                        Paragraph(entity.get('label', ''), cell_style)
                    ])
            else:
                data_for_table.append([
                    Paragraph(f"Sentence {idx + 1}: {sentence_text}", cell_style),
                    Paragraph('No entities found', cell_style),
                    Paragraph('-', cell_style)
                ])

        if data_for_table:
            story.append(Paragraph("<b>Named Entity Recognition</b>", styles['Heading2']))
            story.append(Spacer(1, 12))

            # Adjust column widths (allocate more width to the 'Sentence' column)
            col_widths = [4.5 * inch, 1.5 * inch, 1.5 * inch]

            table = Table(
                [['Sentence', 'Entity', 'Attribute']] + data_for_table,
                colWidths=col_widths,
                repeatRows=1  # Repeat header row on each page
            )
            table.setStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.black),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('WORDWRAP', (0, 0), (-1, -1), 'CJK'),  # Enable word wrap
                ('VALIGN', (0, 0), (-1, -1), 'TOP'),    # Align text to top
                ('FONTSIZE', (0, 0), (-1, -1), 10),     # Adjust font size
                ('LEFTPADDING', (0, 0), (-1, -1), 4),
                ('RIGHTPADDING', (0, 0), (-1, -1), 4),
                ('TOPPADDING', (0, 0), (-1, -1), 4),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
            ])
            story.append(table)
            story.append(Spacer(1, 24))
    except Exception as e:
        print(f"Error adding Named Entity Recognition table: {e}")

    # Build the PDF
    doc.build(story)
    buffer.seek(0)
    return buffer

@app.route("/results/<string:request_id>")
def view_results(request_id):
    """
    Render the analysis results as an HTML page for PDF generation.
    """
    document = collection.find_one({"request_id": request_id})
    if not document or document.get("overall_status") != "processed":
        return "Results not available.", 404

    # Convert ObjectId to string
    document["_id"] = str(document["_id"])

    # Generate absolute URLs for static files
    static_url = url_for('static', filename='', _external=True)

    return render_template("results.html", data=document, static_url=static_url)

@app.route("/send_pdf/<string:request_id>", methods=["GET", "POST"])
def send_pdf(request_id):
    if request.method == "POST":
        email = request.form.get("email")
        if not email:
            return "Email address is required.", 400

        document = collection.find_one({"request_id": request_id})
        if not document or document.get("overall_status") != "processed":
            return "Results not available.", 404

        # generate plots
        images = generate_plots(document)

        # create PDF
        pdf_buffer = create_pdf(document, images)

        # Send email with PDF
        try:
            send_email_with_pdf(email, pdf_buffer.getvalue())
            return render_template("email_sent.html", email=email)
        except Exception as e:
            print(f"Error sending email: {e}")
            return f"Failed to send email: {str(e)}", 500

    return render_template("email_form.html", request_id=request_id)

def send_email_with_pdf(recipient_email, pdf_content
                        , sender_email: str = "garage.swe.team@gmail.com", sender_password: str = "gnjf kukg pqca htnv"):
    """
    Send an email with the PDF attachment to the recipient.
    """
    if not sender_email or not sender_password:
        print("Email credentials are not set.")
        return "Email credentials are not set.", 500

    msg = MIMEMultipart()
    msg["From"] = sender_email
    msg["To"] = recipient_email
    msg["Subject"] = "Your Sentiment Analysis Results"

    # Email body
    body = f"""
Hi {recipient_email.split('@')[0].capitalize()},

This is your sentiment analysis result. Please find attached the PDF of your sentiment analysis results.

Best,

Garage Team
"""
    msg.attach(MIMEText(body, "plain"))

    # Attach PDF
    attachment = MIMEApplication(pdf_content, _subtype="pdf")
    attachment.add_header('Content-Disposition', 'attachment', filename='analysis_results.pdf')
    msg.attach(attachment)

    # Send Email
    try:
        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.ehlo()
            server.starttls()
            server.ehlo()
            server.login(sender_email, sender_password)
            server.send_message(msg)
            print("Email sent successfully.")
    except Exception as e:
        print(f"Failed to send email: {e}")
        raise

if __name__ == "__main__":
    app.run(debug=False, threaded=False)
