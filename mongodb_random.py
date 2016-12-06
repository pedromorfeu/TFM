import numpy as np
from pymongo import MongoClient
from datetime import datetime
import time


MAX = 10


# client = MongoClient("mongodb://pedro2:pedro2@ds053190.mlab.com:53190/")
client = MongoClient("mongodb://localhost:27017")
print(client)
db = client.simulation
print(db)
collection = db.random
print(collection)

res = collection.delete_many({})
print("Deleted", res.deleted_count)

docs = []
for i in range(MAX):
    docs.append( {"number" : np.random.randn()} )

res = collection.insert_many(docs)
print("Inserted", len(res.inserted_ids))

i = 0
numbers = []
for number in collection.find():
    print(i, number["number"], type(number["number"]))
    numbers.append(number["number"])
    i += 1
    time.sleep(0.5)

for number in collection.find(skip=10):
    print(i, number["number"], type(number["number"]))
    numbers.append(number["number"])
    i += 1
    time.sleep(0.5)

client.close()
