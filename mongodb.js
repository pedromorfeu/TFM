db.points.ensureIndex( { "coordinates" : "2dsphere" } )

db.points.deleteMany({})
db.points.insert( { "type" : "Point", "coordinates" : [4,6,0] } )
db.points.insert( { "type" : "Point", "coordinates" : [3,4,1] } )
db.points.find( { "coordinates" : {"$near" : { "$geometry" : { "type" : "point", "coordinates" : [1,2,3] } } } } ).limit(1)

db.points.deleteMany({})
db.points.insert( { "type" : "Point", "coordinates" : [4,6,0,1] } )
db.points.insert( { "type" : "Point", "coordinates" : [3,4,1,4] } )
db.points.find( { "coordinates" : {"$near" : { "$geometry" : { "type" : "point", "coordinates" : [1,2,3,2] } } } } ).limit(1)


docs=[]
for(i=0; i<10000; i++) { docs.push( { "number" : Math.random() } ) }
db.random.insertMany(docs)
db.random.find()
db.random.findOne()
db.random.drop()
