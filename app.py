import os
import io
import sqlite3
from datetime import datetime
from flask import Flask, render_template, request, jsonify, send_file, send_from_directory
import cv2
import numpy as np
import xlsxwriter
from ultralytics import YOLO
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Инициализация модели
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


@app.route('/history', methods=['GET'])
def get_history():
    conn = sqlite3.connect('history.db')
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            'SELECT id, timestamp, result_filename, pedestrian_count '
            'FROM history ORDER BY id DESC'
        ).fetchall()
    finally:
        conn.close()

    return jsonify([
        {
            'id': row['id'],
            'timestamp': row['timestamp'],
            'count': row['pedestrian_count'],
            'result_url': f"/static/uploads/{row['result_filename']}"
        }
        for row in rows
    ])


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

    # Детекция через YOLOv8
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


@app.route('/generate_excel_report/<int:req_id>', methods=['GET'])
def generate_excel_report(req_id):
    conn = sqlite3.connect('history.db')
    try:
        row = conn.execute(
            'SELECT timestamp, result_filename, pedestrian_count FROM history WHERE id = ?',
            (req_id,)
        ).fetchone()
    finally:
        conn.close()

    if not row:
        return jsonify({'error': 'Запись не найдена'}), 404

    timestamp, result_filename, pedestrian_count = row
    result_path = (
        os.path.join(app.config['UPLOAD_FOLDER'], result_filename)
        if result_filename else None
    )
    output = io.BytesIO()
    workbook = xlsxwriter.Workbook(output, {'in_memory': True})
    worksheet = workbook.add_worksheet('Отчёт')
    worksheet.hide_gridlines(2)
    worksheet.set_column('A:A', 30)
    worksheet.set_column('B:B', 28)
    worksheet.set_column('C:C', 3)
    worksheet.set_column('D:D', 18)

    title_format = workbook.add_format({
        'bold': True, 'font_name': 'Arial', 'font_size': 15,
        'font_color': '#17365D', 'bottom': 2, 'bottom_color': '#4472C4'
    })
    label_format = workbook.add_format({
        'bold': True, 'font_name': 'Arial', 'font_color': '#404040',
        'bg_color': '#EAF0F8', 'valign': 'vcenter'
    })
    value_format = workbook.add_format({
        'font_name': 'Arial', 'font_color': '#222222', 'valign': 'vcenter'
    })
    count_format = workbook.add_format({
        'font_name': 'Arial', 'bold': True, 'font_size': 12,
        'font_color': '#217346', 'num_format': '0', 'valign': 'vcenter'
    })
    section_format = workbook.add_format({
        'bold': True, 'font_name': 'Arial', 'font_color': '#17365D',
        'bottom': 1, 'bottom_color': '#B4C7E7'
    })

    worksheet.merge_range('A1:D1', 'Отчёт по анализу пешеходного перехода', title_format)
    worksheet.set_row(0, 26)
    worksheet.write('A3', 'ID запроса', label_format)
    worksheet.write('B3', req_id, value_format)
    worksheet.write('A4', 'Дата и время обработки', label_format)
    worksheet.write('B4', timestamp or '', value_format)
    worksheet.write('A5', 'Обнаружено пешеходов', label_format)
    worksheet.write('B5', pedestrian_count or 0, count_format)
    worksheet.write('A7', 'Изображение с результатом распознавания', section_format)

    image = cv2.imread(result_path) if result_path and os.path.isfile(result_path) else None
    if image is not None:
        height, width = image.shape[:2]
        scale = min(640 / width, 360 / height, 1)
        worksheet.insert_image('A8', result_path, {
            'x_scale': scale,
            'y_scale': scale,
            'object_position': 1,
            'description': 'Изображение с обнаруженными пешеходами'
        })
    else:
        worksheet.write('A8', 'Файл изображения результата недоступен.', value_format)

    workbook.close()
    output.seek(0)
    return send_file(
        output,
        mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        as_attachment=True,
        download_name=f'pedestrian_report_{req_id}.xlsx'
    )


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
