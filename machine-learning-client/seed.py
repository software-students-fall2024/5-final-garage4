"""
This script reads a text file, splits it into sentences, 
and inserts the data into a MongoDB collection.
"""

import os
import uuid
from datetime import datetime

import nltk
from nltk.tokenize import sent_tokenize
from pymongo import MongoClient
from dotenv import load_dotenv

load_dotenv()

nltk.download("punkt")
nltk.download("punkt_tab")

with open("speech.txt", "r", encoding="utf-8") as f:
    speech_text = f.read()

sentences = sent_tokenize(speech_text)
request_id = str(uuid.uuid4())  # pylint: disable=invalid-name
document = {
    "request_id": request_id,
    "sentences": [],
    "overall_status": "pending",
    "timestamp": datetime.now(),
}

for sentence in sentences:
    sentence_entry = {
        "sentence": sentence.strip(),
        "status": "pending",
        "analysis": None,
    }
    document["sentences"].append(sentence_entry)

mongo_uri = os.getenv("MONGO_URI")
client = MongoClient(mongo_uri)
db = client["sentiment"]
texts_collection = db["texts"]

result = texts_collection.insert_one(document)
print(f"Document inserted with _id: {result.inserted_id} and request_id: {request_id}")
