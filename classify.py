import re
import numpy as np
import pandas as pd
import pickle
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import json

# Đường dẫn cơ sở
base_path = r"D:\Chatlocal\pythonProject1\filemodel\3epoch"  # Thay đổi đường dẫn phù hợp với hệ thống của bạn

# Định nghĩa các hàm cần thiết
def preProcess(sentences):
    text = [re.sub(r'([^\s\w]|_)+', '', sentence) for sentence in sentences if sentence != '']
    text = [sentence.lower().strip().split() for sentence in text]
    return text

def load_data_and_tokenizer(data_folder):
    with open(data_folder + r'\data.pkl', 'rb') as file:
        X, y, texts = pickle.load(file)

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(texts)

    return tokenizer

def extract_text_from_json(json_path):
    with open(json_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    # Lấy các đoạn văn bản từ các đối tượng có key 'text'
    text_lines = data['payload']['DETECTION']['text_lines']
    combined_text = " ".join([line['text'] for line in text_lines])

    return [combined_text]  # Trả về dưới dạng danh sách để phù hợp với hàm preProcess

def predict(json_path):
    data_folder = base_path
    model_path = data_folder + r"\3epoch.keras"
    tokenizer = load_data_and_tokenizer(data_folder)

    # Tải mô hình đã huấn luyện
    model = load_model(model_path)

    # Trích xuất và xử lý văn bản từ file JSON
    new_texts = extract_text_from_json(json_path)

    # Xử lý văn bản
    processed_texts = preProcess(new_texts)

    # Biến đổi văn bản thành chuỗi chỉ số
    new_sequences = tokenizer.texts_to_sequences(processed_texts)
    max_len = max(len(seq) for seq in new_sequences)  # Lấy chiều dài lớn nhất từ dữ liệu huấn luyện
    new_sequences = pad_sequences(new_sequences, maxlen=max_len)

    # Dự đoán kết quả
    predictions = model.predict(new_sequences)

    label_mapping = {
        1: "Bao Cao Tai Chinh",
        2: "CCCD Passport",
        3: "Chung tu tai san",
        4: "Hoa don ban hang dien tu",
        5: "Hop dong hon nhan",
        6: "Hop dong lao dong",
        7: "Sao ke"
    }
    predicted_labels = np.argmax(predictions, axis=1)
    predicted_labels_names = [label_mapping[label + 1] for label in predicted_labels]

    # Trả về kết quả dự đoán dưới dạng JSON
    result = []
    for text, label in zip(new_texts, predicted_labels_names):
        result.append({
            "text": text,
            "prediction": label
        })

    return result
