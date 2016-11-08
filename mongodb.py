import numpy
from pymongo import MongoClient
from datetime import datetime


# client = MongoClient("mongodb://pedro2:pedro2@ds053190.mlab.com:53190/")
client = MongoClient("mongodb://localhost:27017")
print(client)
db = client.simulation
print(db)

for i in range(10):
    print("Iteration", i)
    points = 2 * numpy.random.randn(1000000, 3) + 3
    docs = []
    for pi in points:
        x = {"type": "Point", "coordinates": pi.tolist()}
        docs.append(x)
    print(str(datetime.now()), "Inserting", len(docs), "documents")
    db.points.insert_many(docs)
    print(str(datetime.now()), len(docs), "documents inserted")

print(str(datetime.now()))
cursor = db.points.find({"coordinates": {"$near": {"$geometry": {"type": "point", "coordinates": [1, 2, 3]}}}}).limit(1)
print(cursor.next())
print(str(datetime.now()))

client.close()
