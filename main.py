import os
import psycopg2
import faiss
from transformers import AutoTokenizer, AutoModel
import torch
from flask import Flask, request, jsonify
import numpy as np

app = Flask(__name__)

# Настройки для подключения к базе данных PostgreSQL
DB_HOST = os.getenv('DB_HOST', 'localhost')
DB_PORT = os.getenv('DB_PORT', '5432')
DB_NAME = os.getenv('DB_NAME', 'postgres')
DB_USER = os.getenv('DB_USER', 'postgres')
DB_PASS = os.getenv('DB_PASS', '1')
# Установка переменной окружения для OpenMP
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# Загрузка токенизатора и модели
tokenizer = AutoTokenizer.from_pretrained("intfloat/multilingual-e5-large")
model = AutoModel.from_pretrained("intfloat/multilingual-e5-large")


def fetch_texts_from_db():
    connection = None
    texts = {}
    try:
        # Подключение к базе данных
        connection = psycopg2.connect(
            host=DB_HOST,
            port=DB_PORT,
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASS
        )
        cursor = connection.cursor()
        # Выполнение SQL-запроса для получения данных из таблицы items
        cursor.execute("SELECT item_id, item_title, item_description FROM items")
        rows = cursor.fetchall()
        # Извлечение текста из каждой строки результата и создание словаря
        texts = {row[1] + " " + row[2]: row[0] for row in rows}
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
    finally:
        if connection is not None:
            connection.close()
    return texts

texts = fetch_texts_from_db()

def embed_documents(documents_dict, model, tokenizer):
    documents = list(documents_dict.keys())
    inputs = tokenizer(documents, return_tensors='pt', padding=True, truncation=True)
    with torch.no_grad():
        embeddings = model(**inputs).last_hidden_state.mean(dim=1)
    return embeddings, documents

document_embeddings, document_keys = embed_documents(texts, model, tokenizer)
index = faiss.IndexFlatL2(document_embeddings.shape[1])
index.add(document_embeddings.numpy())

def retrieve_documents(query, index, model, tokenizer, texts, top_k=4):
    query_embedding = embed_documents({query: 0}, model, tokenizer)[0].numpy()
    distances, indices = index.search(query_embedding, top_k)
    values = list(texts.values())
    return [values[i] for i in indices[0]]

@app.route('/get_emb', methods=['POST'])
def get_emb():
    data = request.json
    dialog_data = data.get('dialog', [])
    text = " ".join(dialog_data) + ' '

    results = retrieve_documents(text, index, model, tokenizer, texts)
    return jsonify(results)

@app.route('/add_title', methods=['POST'])
def add_title():
    data = request.json
    new_text = data.get('text', None)
    new_id = data.get('id', None)

    texts[new_text] = new_id
    new_embedding = embed_documents({new_text: 0}, model, tokenizer)[0].numpy()
    index.add(new_embedding)

    return jsonify({"message": "Document added successfully"}), 200

@app.route('/delete_title', methods=['POST'])
def delete_title():
    data = request.json
    title_to_delete = data.get('text', None)

    if title_to_delete in texts:
        doc_index = list(texts.keys()).index(title_to_delete)
        del texts[title_to_delete]

        index.remove_ids(np.array([doc_index], dtype=np.int64))

        return jsonify({"message": "Document deleted successfully"}), 200
    else:
        return jsonify({"message": "Document not found"}), 404

if __name__ == '__main__':
    app.run('0.0.0.0', port=5004)
