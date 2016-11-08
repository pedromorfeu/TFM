import numpy
from pymongo import MongoClient
from datetime import datetime


client = MongoClient("mongodb://pedro2:pedro2@ds053190.mlab.com:53190/")
print(client)
db = client.mydb
print(db)
print(db.people.find_one())

points = 2 * numpy.random.randn(10000000, 3) + 3
docs = []
for pi in points:
    x = {"type": "Point", "coordinates": pi}
    docs.append(x)

print(str(datetime.now()), "Inserting documents...")
db.points.insert_many(docs)
print(str(datetime.now()), "Inserted")

print(str(datetime.now()))
res = db.points.find({"coordinates": {"$near": {"$geometry": {"type": "point", "coordinates": [1,2,3,2]}}}}).limit(1)
print(res)
print(str(datetime.now()))

client.close()
