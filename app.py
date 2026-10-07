import os
import sqlite3
from datetime import datetime
from flask import Flask, render_template, request, jsonify, send_file, send_from_directory
import cv2
import numpy as np
from ultralytics import YOLO
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Инициализация модели YOLOv8 (nano — лёгкая и быстрая)
model = YOLO('yolov8n.pt')


# Инициализация базы данных SQLite
def init_db():
    conn = sqlite3.connect('history.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            filename TEXT,
            result_filename TEXT,
            pedestrian_count INTEGER
        )
    ''')
    conn.commit()
    conn.close()


init_db()


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/process', methods=['POST'])
def process_image():
    if 'image' not in request.files:
        return jsonify({'error': 'Файл не найден'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'Файл не выбран'}), 400

    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    orig_filename = f"orig_{timestamp_str}.jpg"
    res_filename = f"res_{timestamp_str}.jpg"

    orig_path = os.path.join(app.config['UPLOAD_FOLDER'], orig_filename)
    res_path = os.path.join(app.config['UPLOAD_FOLDER'], res_filename)

    # Чтение изображения
    file_bytes = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    cv2.imwrite(orig_path, img)

    # Детекция через YOLOv8 (фильтруем класс 0 - person)
    results = model(img, classes=[0])

    # Визуализация рамок
    annotated_frame = results[0].plot()
    cv2.imwrite(res_path, annotated_frame)

    # Подсчет пешеходов
    count = len(results[0].boxes)
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Сохранение в БД
    conn = sqlite3.connect('history.db')
    cursor = conn.cursor()
    cursor.execute(
        'INSERT INTO history (timestamp, filename, result_filename, pedestrian_count) VALUES (?, ?, ?, ?)',
        (now_str, orig_filename, res_filename, count)
    )
    request_id = cursor.lastrowid
    conn.commit()
    conn.close()

    return jsonify({
        'id': request_id,
        'count': count,
        'result_url': f'/static/uploads/{res_filename}',
        'timestamp': now_str
    })


@app.route('/generate_report/<int:req_id>', methods=['GET'])
def generate_report(req_id):
    conn = sqlite3.connect('history.db')
    cursor = conn.cursor()
    cursor.execute('SELECT timestamp, result_filename, pedestrian_count FROM history WHERE id = ?', (req_id,))
    row = cursor.fetchone()
    conn.close()

    if not row:
        return jsonify({'error': 'Запись не найдена'}), 404

    ts, res_filename, count = row
    pdf_path = os.path.join(app.config['UPLOAD_FOLDER'], f"report_{req_id}.pdf")
    img_path = os.path.join(app.config['UPLOAD_FOLDER'], res_filename)

    # Генерация PDF-отчета c помощью ReportLab
    c = canvas.Canvas(pdf_path, pagesize=letter)
    c.setFont("Helvetica-Bold", 18)
    c.drawString(50, 750, "Отчет по анализу пешеходного перехода")

    c.setFont("Helvetica", 12)
    c.drawString(50, 720, f"ID запроса: {req_id}")
    c.drawString(50, 700, f"Дата и время обработки: {ts}")
    c.drawString(50, 680, f"Количество обнаруженных пешеходов: {count}")

    if os.path.exists(img_path):
        c.drawImage(img_path, 50, 350, width=400, height=300)

    c.save()
    return send_file(pdf_path, as_attachment=True, download_name=f"report_{req_id}.pdf")


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)