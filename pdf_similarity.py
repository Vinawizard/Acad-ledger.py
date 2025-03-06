import os
import fitz
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = {"pdf"}

# Ensure upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Extract text from PDF
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text("text") + "\n"
    return text.strip()

# Process PDFs in a directory
def process_pdfs(pdf_folder):
    pdf_texts = {}
    for filename in os.listdir(pdf_folder):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(pdf_folder, filename)
            text = extract_text_from_pdf(pdf_path)
            pdf_texts[filename] = text
    return pdf_texts

# Compute similarity
def compute_similarity(pdf_texts, new_pdf_text):
    all_texts = list(pdf_texts.values()) + [new_pdf_text]
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(all_texts)
    similarity_scores = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1]).flatten()
    results_df = pd.DataFrame({
        "PDF Name": list(pdf_texts.keys()),
        "Similarity Score": similarity_scores
    }).sort_values(by="Similarity Score", ascending=False)
    return results_df

# Check if file is allowed
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

# Home route - Serve the upload page
@app.route("/", methods=["GET", "POST"])
def upload_file():
    if request.method == "POST":
        # Check if files are uploaded
        if "new_pdf" not in request.files or "compare_pdfs" not in request.files:
            return jsonify({"error": "Missing files"}), 400
        
        new_pdf = request.files["new_pdf"]
        compare_pdfs = request.files.getlist("compare_pdfs")

        if new_pdf and allowed_file(new_pdf.filename):
            # Save the new PDF
            new_pdf_filename = secure_filename(new_pdf.filename)
            new_pdf_path = os.path.join(app.config["UPLOAD_FOLDER"], new_pdf_filename)
            new_pdf.save(new_pdf_path)

            # Save comparison PDFs to a temp folder
            compare_folder = os.path.join(app.config["UPLOAD_FOLDER"], "compare")
            if not os.path.exists(compare_folder):
                os.makedirs(compare_folder)
            
            for pdf in compare_pdfs:
                if allowed_file(pdf.filename):
                    pdf_filename = secure_filename(pdf.filename)
                    pdf.save(os.path.join(compare_folder, pdf_filename))

            # Process and compute similarity
            pdf_texts = process_pdfs(compare_folder)
            new_pdf_text = extract_text_from_pdf(new_pdf_path)
            results_df = compute_similarity(pdf_texts, new_pdf_text)

            # Convert results to list of dicts for JSON
            results = results_df.to_dict(orient="records")
            return jsonify({"results": results})

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)