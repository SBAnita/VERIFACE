import os
import io
import threading
import sqlite3
import datetime
import json
from flask import Flask, redirect, render_template, request, jsonify, send_file, abort
from model import train_model_background, extract_embedding_for_image, MODEL_PATH

APP_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(APP_DIR, "attendance.db")
DATASET_DIR = os.path.join(APP_DIR, "dataset")
os.makedirs(DATASET_DIR, exist_ok=True)

TRAIN_STATUS_FILE = os.path.join(APP_DIR, "train_status.json")

app = Flask(__name__, static_folder="static", template_folder="templates")

# ---------- DB helpers ----------
def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""CREATE TABLE IF NOT EXISTS students (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    roll TEXT,
                    class TEXT,
                    section TEXT,
                    reg_no TEXT,
                    created_at TEXT
                )""")
    c.execute("""CREATE TABLE IF NOT EXISTS attendance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    student_id INTEGER,
                    name TEXT,
                    timestamp TEXT
                )""")
    conn.commit()
    conn.close()

init_db()

# ---------- Train status helpers ----------
def write_train_status(status_dict):
    with open(TRAIN_STATUS_FILE, "w") as f:
        json.dump(status_dict, f)

def read_train_status():
    if not os.path.exists(TRAIN_STATUS_FILE):
        return {"running": False, "progress": 0, "message": "Not trained"}
    with open(TRAIN_STATUS_FILE, "r") as f:
        return json.load(f)

# ensure initial train status file exists
write_train_status({"running": False, "progress": 0, "message": "No training yet."})

# ---------- Routes ----------
@app.route("/")
def main():
    return render_template("main.html")

@app.route("/dashboard")
def dashboard():
    return render_template("/index.html")

# Dashboard simple API for attendance stats (last 30 days)
@app.route("/attendance_stats")
def attendance_stats():
    import pandas as pd
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT timestamp FROM attendance", conn)
    conn.close()
    if df.empty:
        from datetime import date, timedelta
        days = [(date.today() - datetime.timedelta(days=i)).strftime("%d-%b") for i in range(29, -1, -1)]
        return jsonify({"dates": days, "counts": [0]*30})
    df['date'] = pd.to_datetime(df['timestamp']).dt.date
    last_30 = [ (datetime.date.today() - datetime.timedelta(days=i)) for i in range(29, -1, -1) ]
    counts = [ int(df[df['date'] == d].shape[0]) for d in last_30 ]
    dates = [ d.strftime("%d-%b") for d in last_30 ]
    return jsonify({"dates": dates, "counts": counts})

# -------- Add student (form) --------
@app.route("/add_student", methods=["GET", "POST"])
def add_student():
    if request.method == "GET":
        return render_template("add_student.html")
    # POST: save student metadata and return student_id
    data = request.form
    name = data.get("name","").strip()
    roll = data.get("roll","").strip()
    cls = data.get("class","").strip()
    sec = data.get("sec","").strip()
    reg_no = data.get("reg_no","").strip()
    if not name:
        return jsonify({"error":"name required"}), 400
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    now = datetime.datetime.utcnow().isoformat()
    c.execute("INSERT INTO students (name, roll, class, section, reg_no, created_at) VALUES (?, ?, ?, ?, ?, ?)",
              (name, roll, cls, sec, reg_no, now))
    sid = c.lastrowid
    conn.commit()
    conn.close()
    # create dataset folder for this student
    os.makedirs(os.path.join(DATASET_DIR, str(sid)), exist_ok=True)
    return jsonify({"student_id": sid})

# -------- Show student details --------
@app.route("/student_details")
def student_details():

    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    c.execute("""
        SELECT
            id,
            name,
            roll,
            reg_no,
            class,
            section,
            created_at
        FROM students
        ORDER BY id DESC
    """)

    students = c.fetchall()

    conn.close()

    return render_template(
        "student_details.html",
        students=students
    )

# -------- Edit student details --------
@app.route("/edit_student/<int:sid>", methods=["GET", "POST"])
def edit_student(sid):

    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    if request.method == "POST":

        name = request.form["name"]
        roll = request.form["roll"]
        reg_no = request.form["reg_no"]
        cls = request.form["class"]
        section = request.form["section"]

        c.execute("""
            UPDATE students
            SET
                name=?,
                roll=?,
                reg_no=?,
                class=?,
                section=?
            WHERE id=?
        """, (
            name,
            roll,
            reg_no,
            cls,
            section,
            sid
        ))

        conn.commit()
        conn.close()

        return redirect("/student_details")

    c.execute("""
        SELECT
            id,
            name,
            roll,
            reg_no,
            class,
            section
        FROM students
        WHERE id=?
    """, (sid,))

    student = c.fetchone()

    conn.close()

    return render_template(
        "edit_student.html",
        student=student
    )

# -------- Delete student details --------
@app.route("/delete_student/<int:sid>")
def delete_student_page(sid):

    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    c.execute(
        "DELETE FROM attendance WHERE student_id=?",
        (sid,)
    )

    c.execute(
        "DELETE FROM students WHERE id=?",
        (sid,)
    )

    conn.commit()
    conn.close()

    return redirect("/student_details")

# -------- Upload face images (after capture) --------
@app.route("/upload_face", methods=["POST"])
def upload_face():
    student_id = request.form.get("student_id")
    if not student_id:
        return jsonify({"error":"student_id required"}), 400
    files = request.files.getlist("images[]")
    saved = 0
    folder = os.path.join(DATASET_DIR, student_id)
    if not os.path.isdir(folder):
        os.makedirs(folder, exist_ok=True)
    for f in files:
        try:
            fname = f"{datetime.datetime.utcnow().timestamp():.6f}_{saved}.jpg"
            path = os.path.join(folder, fname)
            f.save(path)
            saved += 1
        except Exception as e:
            app.logger.error("save error: %s", e)
    return jsonify({"saved": saved})

# -------- Train model (start background thread) --------
@app.route("/train_model", methods=["GET"])
def train_model_route():
    # if already running, respond accordingly
    status = read_train_status()
    if status.get("running"):
        return jsonify({"status":"already_running"}), 202
    # reset status
    write_train_status({"running": True, "progress": 0, "message": "Starting training"})
    # start background thread
    t = threading.Thread(target=train_model_background, args=(DATASET_DIR, lambda p,m: write_train_status({"running": True, "progress": p, "message": m})))
    t.daemon = True
    t.start()
    return jsonify({"status":"started"}), 202

# -------- Train progress (polling) --------
@app.route("/train_status", methods=["GET"])
def train_status():
    return jsonify(read_train_status())

# ================= STUDENTS =================
@app.route("/students")
def students():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        SELECT id, name, roll, reg_no, class, section, created_at
        FROM students
        ORDER BY id DESC
    """)
    students = c.fetchall()
    conn.close()
    return render_template("students.html", students=students)

# -------- Mark attendance page --------
@app.route("/mark_attendance", methods=["GET"])
def mark_attendance_page():
    return render_template("mark_attendance.html")

# -------- Recognize face endpoint (POST image) --------
@app.route("/recognize_face", methods=["POST"])
def recognize_face():
    if "image" not in request.files:
        return jsonify({"recognized": False, "error": "no image"}), 400

    img_file = request.files["image"]

    try:
        emb = extract_embedding_for_image(img_file.stream)

        if emb is None:
            return jsonify({"recognized": False, "error": "no face detected"}), 200

        from model import load_model_if_exists, predict_with_model

        clf = load_model_if_exists()

        if clf is None:
            return jsonify({"recognized": False, "error": "model not trained"}), 200

        pred_label, conf = predict_with_model(clf, emb)

        if conf < 0.5:
            return jsonify({
                "recognized": False,
                "confidence": float(conf)
            }), 200

        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()

        # Get student name
        c.execute(
            "SELECT name FROM students WHERE id=?",
            (int(pred_label),)
        )

        row = c.fetchone()
        name = row[0] if row else "Unknown"

        # Check if already marked today
        today = datetime.date.today().isoformat()

        c.execute("""
            SELECT id
            FROM attendance
            WHERE student_id=?
            AND DATE(timestamp)=?
        """, (
            int(pred_label),
            today
        ))

        already_marked = c.fetchone()

        # Insert only once per day
        if already_marked is None:

            ts = datetime.datetime.utcnow().isoformat()

            c.execute("""
                INSERT INTO attendance
                (student_id, name, timestamp)
                VALUES (?, ?, ?)
            """, (
                int(pred_label),
                name,
                ts
            ))

            conn.commit()

        conn.close()

        return jsonify({
            "recognized": True,
            "student_id": int(pred_label),
            "name": name,
            "confidence": float(conf)
        }), 200

    except Exception as e:
        app.logger.exception("recognize error")

        return jsonify({
            "recognized": False,
            "error": str(e)
        }), 500

# -------- Attendance records & filters --------
@app.route("/attendance_record", methods=["GET"])
def attendance_record():

    period = request.args.get("period", "daily")

    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    # DAILY REPORT
    if period == "daily":

        selected_date = request.args.get(
            "selected_date",
            datetime.date.today().isoformat()
        )

        c.execute("""
            SELECT
                MIN(id),
                student_id,
                name,
                MIN(timestamp)
            FROM attendance
            WHERE DATE(timestamp)=?
            GROUP BY student_id, name
            ORDER BY name
        """, (selected_date,))

        records = c.fetchall()

        c.execute("""
            SELECT COUNT(DISTINCT student_id)
            FROM attendance
            WHERE DATE(timestamp)=?
        """, (selected_date,))

        total_present = c.fetchone()[0]

        conn.close()

        return render_template(
            "attendance_record.html",
            period="daily",
            records=records,
            selected_date=selected_date,
            total_present=total_present
        )

    # MONTHLY REPORT

    month = request.args.get(
        "month",
        datetime.date.today().strftime("%Y-%m")
    )

    c.execute("""
        SELECT id,name
        FROM students
        ORDER BY name
    """)

    students = c.fetchall()

    # Working days in selected month
    c.execute("""
        SELECT COUNT(DISTINCT DATE(timestamp))
        FROM attendance
        WHERE strftime('%Y-%m', timestamp)=?
    """, (month,))

    working_days = c.fetchone()[0]

    if working_days == 0:
        working_days = 1

    monthly_data = []

    for student in students:

        sid = student[0]
        name = student[1]

        c.execute("""
            SELECT COUNT(DISTINCT DATE(timestamp))
            FROM attendance
            WHERE student_id=?
            AND strftime('%Y-%m', timestamp)=?
        """, (sid, month))

        present_days = c.fetchone()[0]

        percentage = round(
            (present_days / working_days) * 100,
            2
        )

        monthly_data.append({
            "name": name,
            "present_days": present_days,
            "total_days": working_days,
            "percentage": percentage
        })

    conn.close()

    return render_template(
        "attendance_record.html",
        period="monthly",
        monthly_data=monthly_data,
        month=month
    )

@app.route("/attendance/<int:aid>", methods=["DELETE"])
def delete_attendance(aid):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("DELETE FROM attendance WHERE id=?", (aid,))
    conn.commit()
    conn.close()
    return jsonify({"deleted": True})

# -------- CSV download --------
@app.route("/download_csv")
def download_csv():

    period = request.args.get("period", "daily")

    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    output = io.StringIO()

    # ---------------- DAILY CSV ----------------
    if period == "daily":

        selected_date = request.args.get(
            "selected_date",
            datetime.date.today().isoformat()
        )

        c.execute("""
            SELECT
                student_id,
                name,
                MIN(timestamp)
            FROM attendance
            WHERE DATE(timestamp)=?
            GROUP BY student_id, name
            ORDER BY name
        """, (selected_date,))

        records = c.fetchall()

        total_present = len(records)

        output.write(f"Total Present:,{total_present}\n\n")
        output.write("S.No,Name,Time\n")

        for i, r in enumerate(records, start=1):
            output.write(
                f"{i},{r[1]},{r[2]}\n"
            )

        filename = f"daily_attendance_{selected_date}.csv"

    # ---------------- MONTHLY CSV ----------------
    else:

        month = request.args.get(
            "month",
            datetime.date.today().strftime("%Y-%m")
        )

        c.execute("""
            SELECT id,name
            FROM students
            ORDER BY name
        """)

        students = c.fetchall()

        c.execute("""
            SELECT COUNT(DISTINCT DATE(timestamp))
            FROM attendance
            WHERE strftime('%Y-%m', timestamp)=?
        """, (month,))

        working_days = c.fetchone()[0]

        if working_days == 0:
            working_days = 1

        output.write(
            "S.No,Student Name,Present Days,Total Working Days,Attendance %,Status\n"
        )

        for i, student in enumerate(students, start=1):

            sid = student[0]
            name = student[1]

            c.execute("""
                SELECT COUNT(DISTINCT DATE(timestamp))
                FROM attendance
                WHERE student_id=?
                AND strftime('%Y-%m', timestamp)=?
            """, (
                sid,
                month
            ))

            present_days = c.fetchone()[0]

            percentage = round(
                (present_days / working_days) * 100,
                2
            )

            status = (
                "Eligible"
                if percentage >= 75
                else "Below 75%"
            )

            output.write(
                f"{i},{name},{present_days},{working_days},{percentage}%,{status}\n"
            )

        filename = f"monthly_attendance_{month}.csv"

    conn.close()

    mem = io.BytesIO()
    mem.write(output.getvalue().encode("utf-8"))
    mem.seek(0)

    return send_file(
        mem,
        as_attachment=True,
        download_name=filename,
        mimetype="text/csv"
    )

# -------- Download Students CSV --------
@app.route("/download_students_csv")
def download_students_csv():

    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    c.execute("""
        SELECT
            name,
            roll,
            reg_no,
            class,
            section,
            created_at
        FROM students
        ORDER BY name
    """)

    students = c.fetchall()

    conn.close()

    output = io.StringIO()

    output.write(
        "S.No,Name,Roll,Reg No,Class,Section,Created\n"
    )

    for i, s in enumerate(students, start=1):

        output.write(
            f"{i},{s[0]},{s[1]},{s[2]},{s[3]},{s[4]},{s[5]}\n"
        )

    mem = io.BytesIO()
    mem.write(output.getvalue().encode("utf-8"))
    mem.seek(0)

    return send_file(
        mem,
        as_attachment=True,
        download_name="student_details.csv",
        mimetype="text/csv"
    )

# ---------------- run ------------------------
if __name__ == "__main__":
    app.run(debug=True)