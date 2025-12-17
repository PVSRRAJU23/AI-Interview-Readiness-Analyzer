from flask import Flask, request, jsonify, render_template, send_file
import pickle
from preprocess import clean_text
from sklearn.metrics.pairwise import cosine_similarity
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
import io

app = Flask(__name__)

# ---------- LOAD ML COMPONENTS ----------
model = pickle.load(open("models/model.pkl", "rb"))
vectorizer = pickle.load(open("models/vectorizer.pkl", "rb"))
encoder = pickle.load(open("models/encoder.pkl", "rb"))

# ---------- ROLE-BASED REQUIRED SKILLS ----------
REQUIRED_SKILLS = {
    "aiml": [
        "python",
        "machine learning",
        "deep learning",
        "sql",
        "statistics"
    ],
    "data_scientist": [
        "python",
        "statistics",
        "data analysis",
        "sql",
        "machine learning"
    ],
    "software_engineer": [
        "python",
        "data structures",
        "algorithms",
        "sql",
        "system design"
    ]
}

# ---------- HOME ----------
@app.route("/")
def home():
    return render_template("index.html")


# ---------- ANALYZE (FIXED LOGIC) ----------
@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.get_json()

    if not data or "resume" not in data:
        return jsonify({"error": "resume field missing"}), 400

    # Clean inputs
    resume = clean_text(data["resume"])
    jd = clean_text(data.get("job_description", ""))
    role = data.get("role", "aiml")

    # ---------- REQUIRED SKILLS ----------
    required = REQUIRED_SKILLS.get(role, REQUIRED_SKILLS["aiml"])

    # ---------- SKILL MATCH ----------
    found_skills = [skill for skill in required if skill in resume]
    missing_skills = [skill for skill in required if skill not in resume]

    skill_match_percent = round(
        (len(found_skills) / len(required)) * 100, 2
    )

    # ---------- JD MATCH ----------
    resume_vec = vectorizer.transform([resume])
    if jd:
        jd_vec = vectorizer.transform([jd])
        jd_match_percent = round(
            cosine_similarity(resume_vec, jd_vec)[0][0] * 100, 2
        )
    else:
        jd_match_percent = 0

    # ---------- ML PREDICTION (LIGHT WEIGHT) ----------
    prediction = model.predict(resume_vec)
    ml_label = encoder.inverse_transform(prediction)[0]

    ml_score = 90 if ml_label == "ready" else 60 if ml_label == "partially_ready" else 30

    # ---------- FINAL READINESS SCORE (HYBRID) ----------
    final_score = round(
        0.5 * skill_match_percent +
        0.3 * jd_match_percent +
        0.2 * ml_score,
        2
    )

    # ---------- READINESS LABEL ----------
    if final_score >= 75:
        readiness = "ready"
    elif final_score >= 50:
        readiness = "partially_ready"
    else:
        readiness = "not_ready"

    # ---------- RECOMMENDATIONS ----------
    recommendations = [
        f"Improve your knowledge in {skill}"
        for skill in missing_skills
    ]

    return jsonify({
        "selected_role": role,
        "interview_readiness": readiness,
        "readiness_score": final_score,
        "skill_match_percentage": skill_match_percent,
        "jd_match_percentage": jd_match_percent,
        "missing_skills": missing_skills,
        "recommendations": recommendations
    })


# ---------- PDF REPORT ----------
@app.route("/download-report", methods=["POST"])
def download_report():
    data = request.get_json()

    buffer = io.BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=A4)
    pdf.setFont("Helvetica", 12)

    pdf.drawString(50, 800, "AI Interview Readiness Report")
    pdf.drawString(50, 770, f"Role: {data['selected_role']}")
    pdf.drawString(50, 750, f"Readiness: {data['interview_readiness']}")
    pdf.drawString(50, 730, f"Score: {data['readiness_score']}/100")
    pdf.drawString(50, 710, f"Skill Match: {data['skill_match_percentage']}%")
    pdf.drawString(50, 690, f"JD Match: {data['jd_match_percentage']}%")

    pdf.drawString(50, 660, "Missing Skills:")
    y = 640
    for skill in data["missing_skills"]:
        pdf.drawString(70, y, f"- {skill}")
        y -= 18

    pdf.drawString(50, y - 20, "Recommendations:")
    y -= 40
    for rec in data["recommendations"]:
        pdf.drawString(70, y, f"- {rec}")
        y -= 18

    pdf.showPage()
    pdf.save()
    buffer.seek(0)

    return send_file(
        buffer,
        as_attachment=True,
        download_name="Interview_Readiness_Report.pdf",
        mimetype="application/pdf"
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
