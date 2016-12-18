from pymongo import MongoClient

print("Util :: Mongo")

client = MongoClient("mongodb://localhost:27017")
print(client)
db = client.simulation
print(db)
data_collection = db.data
print(data_collection)
component_collection = db.component
print(component_collection)
generated_collection = db.generated
print(generated_collection)


# Clear collections
def clear_all():
    data_collection.delete_many({})
    component_collection.delete_many({})
    generated_collection.delete_many({})


def store_data(matrix, schema, type):
    store_many(matrix, schema, data_collection, type)


def store_component(matrix, schema, type):
    store_many(matrix, schema, component_collection, type)


def store_generated(matrix, schema, type):
    store_many(matrix, schema, generated_collection, type)


def store_many(matrix, schema, collection, type):
    # Store components values in MongoDB
    docs_component = []
    for observation in matrix:
        obs = ([type] + observation.tolist()) if (type is not None) else observation.tolist()
        doc_component = dict(zip(schema, obs))
        docs_component.append(doc_component)
    collection.insert_many(docs_component)


def store_generated_single(doc):
    generated_collection.insert_one(doc)
