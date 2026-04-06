# test_connection.py
from pymongo import MongoClient
import os

# Your connection string (keep it in .env in real projects!)
# uri = "mongodb+srv://muhammad:blyat925@cluster0.bxrzccu.mongodb.net/?appName=Cluster0"
uri = os.getenv("MONGODB_URI")

client = MongoClient(uri)

# Simple ping test
client = MongoClient(uri)
db = client["myapp"]  # auto-created
collection = db["notes"]  # auto-created

# Insert something
result = collection.insert_one(
    {
        "title": "Second note from uv project",
        "content": "uv is super slow!",
        "tags": ["js", "mongodb", "Karachi"],
        "city": "Karachi",
    }
)

print("Inserted ID:", result.inserted_id)

# Quick read
print("\nAll notes:")
for note in collection.find().limit(5):
    print(note)
