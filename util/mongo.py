from pymongo import MongoClient

print("Util :: Mongo")


def store(matrix, schema, collection, type=None):
    # Store components values in MongoDB
    docs_component = []
    for observation in matrix:
        obs = [type] + observation.tolist()
        if type is None:
            obs = observation.tolist()
        doc_component = dict(zip(schema, obs))
        docs_component.append(doc_component)
    collection.insert_many(docs_component)
