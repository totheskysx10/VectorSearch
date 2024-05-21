import os
import psycopg2
import faiss
from transformers import AutoTokenizer, AutoModel
import torch
from flask import Flask, request, jsonify

app = Flask(__name__)

# Настройки для подключения к базе данных PostgreSQL
DB_HOST = "localhost"
DB_PORT = "5432"
DB_NAME = "postgres"
DB_USER = "postgres"
DB_PASS = "1"

# Установка переменной окружения для OpenMP
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# Загрузка токенизатора и модели
tokenizer = AutoTokenizer.from_pretrained("intfloat/multilingual-e5-large")
model = AutoModel.from_pretrained("intfloat/multilingual-e5-large")

def fetch_texts_from_db():
    connection = None
    texts = []
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
        cursor.execute("SELECT item_title FROM items")
        rows = cursor.fetchall()
        # Извлечение текста из каждой строки результата
        texts = [row[0] for row in rows]
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
    finally:
        if connection is not None:
            connection.close()
    return texts

texts = fetch_texts_from_db()

def embed_documents(documents, model, tokenizer):
    inputs = tokenizer(documents, return_tensors='pt', padding=True, truncation=True)
    with torch.no_grad():
        embeddings = model(**inputs).last_hidden_state.mean(dim=1)
    return embeddings

document_embeddings = embed_documents(texts, model, tokenizer)
index = faiss.IndexFlatL2(document_embeddings.shape[1])
index.add(document_embeddings.numpy())

def retrieve_documents(query, index, model, tokenizer, texts, top_k=2):
    query_embedding = embed_documents([query], model, tokenizer).numpy()
    distances, indices = index.search(query_embedding, top_k)
    return [texts[i] for i in indices[0]]

@app.route('/get_emb', methods=['POST'])
def get_emb():
    data = request.json
    dialog_data = data.get('dialog', [])
    text = " ".join([s for s in dialog_data])

    results = retrieve_documents(text, index, model, tokenizer, texts)
    return jsonify(results)

@app.route('/add_title', methods=['POST'])
def add_title():
    data = request.json
    new_text = data.get('text', None)

    texts.append(new_text)
    new_embedding = embed_documents([new_text], model, tokenizer).numpy()
    index.add(new_embedding)

    return jsonify({"message": "Document added successfully"}), 200


@app.route('/delete_title', methods=['POST'])
def delete_title():
    data = request.json
    title_to_delete = data.get('text', None)

    if title_to_delete in texts:
        texts.remove(title_to_delete)

        document_embeddings = embed_documents(texts, model, tokenizer)

        index.reset()
        index.add(document_embeddings.numpy())

        return jsonify({"message": "Document deleted successfully"}), 200
    else:
        return jsonify({"message": "Document not found"}), 404


if __name__ == '__main__':
    app.run('0.0.0.0', port=5004)
