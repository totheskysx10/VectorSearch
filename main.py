import os
import psycopg2
import torch
from flask import Flask, request, jsonify
from qdrant_client.http.models import VectorParams, Distance
from transformers import AutoTokenizer, AutoModel
from qdrant_client import QdrantClient, models
from qdrant_client.http import models as rest
import gc

app = Flask(__name__)

# Настройки для подключения к базе данных PostgreSQL
DB_HOST = os.getenv('DB_HOST', 'localhost')
DB_PORT = os.getenv('DB_PORT', '5432')
DB_NAME = os.getenv('DB_NAME', 'postgres')
DB_USER = os.getenv('DB_USER', 'postgres')
DB_PASS = os.getenv('DB_PASS', '1')
# Установка переменной окружения для OpenMP
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# Настройки для подключения к Qdrant
QDRANT_HOST = os.getenv('QDRANT_HOST', 'localhost')
QDRANT_PORT = os.getenv('QDRANT_PORT', 6333)

# Загрузка токенизатора и модели
tokenizer = AutoTokenizer.from_pretrained("intfloat/multilingual-e5-large")
model = AutoModel.from_pretrained("intfloat/multilingual-e5-large")

# Инициализация клиента Qdrant
qdrant_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

# Название коллекции Qdrant
COLLECTION_NAME = 'item_embeddings'


def fetch_texts_from_db(batch_size=10):
    connection = None
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
        # Выполнение SQL-запроса для получения данных из таблицы items с категориями
        cursor.execute("""
            SELECT i.item_id, i.item_title, i.item_description, c.category_title
            FROM items i
            LEFT JOIN categories c ON i.category_id = c.category_id
        """)
        while True:
            rows = cursor.fetchmany(batch_size)
            if not rows:
                break
            texts = {}
            for row in rows:
                item_id = row[0]
                item_title = row[1]
                item_description = row[2]
                category_title = row[3]

                if category_title is None:
                    text_key = f"{item_title} {item_description}"
                else:
                    text_key = f"{item_title} {category_title} {item_description}"

                texts[text_key] = item_id

            yield texts
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
    finally:
        if connection is not None:
            connection.close()

def embed_documents(documents_dict, model, tokenizer):
    documents = list(documents_dict.keys())
    inputs = tokenizer(documents, return_tensors='pt', padding=True, truncation=True)
    with torch.no_grad():
        embeddings = model(**inputs).last_hidden_state.mean(dim=1)
    return embeddings, documents

def add_embeddings_to_qdrant(texts, model, tokenizer):
    embeddings, documents = embed_documents(texts, model, tokenizer)
    for doc, embedding in zip(documents, embeddings):
        qdrant_client.upsert(
            collection_name=COLLECTION_NAME,
            points=[rest.PointStruct(id=texts[doc], vector=embedding.numpy().tolist(), payload={"text": doc})]
        )

def sync_embeddings(texts, model, tokenizer):
    points, _ = qdrant_client.scroll(collection_name=COLLECTION_NAME, limit=10000)
    existing_texts = set(point.payload["text"] for point in points)

    texts_to_remove = existing_texts - set(texts.keys())

    if texts_to_remove:
        points_to_remove = [point.id for point in points if point.payload['text'] in texts_to_remove]
        qdrant_client.delete(collection_name=COLLECTION_NAME, points_selector=points_to_remove)

    new_texts = {text: texts[text] for text in texts if text not in existing_texts}
    if new_texts:
        add_embeddings_to_qdrant(new_texts, model, tokenizer)


def create_qdrant_collection():
    collections = qdrant_client.get_collections()
    collections_list = [c.name for collection in collections for c in list(collection[1])]
    if COLLECTION_NAME not in collections_list:
        qdrant_client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=1024, distance=Distance.COSINE)
        )
    texts_generator = fetch_texts_from_db()
    all_texts = {}
    for texts in texts_generator:
        all_texts.update(texts)
    sync_embeddings(all_texts, model, tokenizer)
    del all_texts
    gc.collect()

def retrieve_documents(query, qdrant_client, model, tokenizer, top_k=4):
    query_embedding = embed_documents({query: 0}, model, tokenizer)[0].squeeze(0).tolist() # Преобразование в список чисел
    search_result = qdrant_client.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_embedding,
        limit=top_k
    )

    return [item.id for item in search_result]


@app.route('/get_emb', methods=['POST'])
def get_emb():
    data = request.json
    dialog_data = data.get('dialog', [])
    text = " ".join(dialog_data) + ' '

    results = retrieve_documents(text, qdrant_client, model, tokenizer)
    return jsonify(results)


@app.route('/add_title', methods=['POST'])
def add_title():
    data = request.json
    new_text = data.get('text', None)
    new_id = data.get('id', None)

    if new_text and new_id:
        new_embedding = embed_documents({new_text: 0}, model, tokenizer)[0].squeeze(0).tolist()
        qdrant_client.upsert(
            collection_name=COLLECTION_NAME,
            points=[rest.PointStruct(id=new_id, vector=new_embedding, payload={"text": new_text})]
        )
        return jsonify({"message": "Document added successfully"}), 200
    else:
        return jsonify({"message": "Invalid input"}), 400


@app.route('/delete_title', methods=['POST'])
def delete_title():
    data = request.json
    title_to_delete = data.get('text', None)

    qdrant_client.delete(
        collection_name=COLLECTION_NAME,
        points_selector=models.FilterSelector(
            filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="text",
                        match=models.MatchValue(value=title_to_delete),
                    ),
                ],
            )
        ),
    )
    return jsonify({"message": "Document deleted successfully"}), 200

@app.route('/sync_database', methods=['POST'])
def sync_database():
    texts_generator = fetch_texts_from_db()
    all_texts = {}
    for texts in texts_generator:
        all_texts.update(texts)
    sync_embeddings(all_texts, model, tokenizer)
    del all_texts
    gc.collect()
    return jsonify({"message": "Data sync successfully"}), 200

if __name__ == '__main__':
    create_qdrant_collection()
    app.run('0.0.0.0', port=5004)
