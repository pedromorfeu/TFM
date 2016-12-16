import numpy
import pymongo
from pymongo import MongoClient
from datetime import datetime


# client = MongoClient("mongodb://pedro2:pedro2@ds053190.mlab.com:53190/")
client = MongoClient("mongodb://localhost:27017")
print(client)
db = client.simulation
print(db)
collection = db.generated
print(collection)

for doc in collection.find({"type": "inverse"}).sort("_id", pymongo.ASCENDING):
    print(doc["APHu"])

print("---")
print(collection.find_one({"type": "inverse"}, sort=[("_id", pymongo.ASCENDING)], skip=12))

exit()

# db.points.delete_many({})

points = 2 * numpy.random.randn(10000000, 3) + 3

init = 0
step = 1000000
for i in range(10):
    print("Iteration", i)
    points_sample = points[init:init+step]
    init += step
    docs = []
    for pi in points_sample:
        x = {"type": "Point", "coordinates": pi.tolist()}
        docs.append(x)
    print(str(datetime.now()), "Inserting", len(docs), "documents")
    db.points.insert_many(docs)
    print(str(datetime.now()), len(docs), "documents inserted")


# MongoDB distance
# A 2dsphere index in MongoDB 3.0+ will allow additional values in the coordinate array,
# but only the first two values will be indexed and used in a 2dsphere query
print(str(datetime.now()))
cursor = db.points.find({"coordinates": {"$near": {"$geometry": {"type": "point", "coordinates": [1, 2, 3]}}}}).limit(1)
print(cursor.next())
print(str(datetime.now()))


# Python distance
p = (1, 2, 3)
print(str(datetime.now()))
distances = numpy.sqrt(((points - p)**2).sum(axis=1))
sorted_indexes = distances.argsort()
print(points[sorted_indexes][0])
print(str(datetime.now()))


client.close()
