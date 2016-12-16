import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pymongo
from pymongo import MongoClient

client = MongoClient("mongodb://localhost:27017")
print(client)
db = client.simulation
print(db)
collection = db.generated
print(collection)


def data_gen(t=0):
    # loop for each new data point
    print("data_gen", t)
    while True:
        doc = collection.find_one({"type": "inverse"}, sort=[("_id", pymongo.ASCENDING)], skip=t)
        if doc is not None:
            t += 1
            yield t, doc["APHu"]
        else:
            yield t, None


def init():
    ax.set_ylim(min(ydata), max(ydata))
    ax.set_xlim(0, len(xdata))
    # del xdata[:]
    # del ydata[:]
    print("init", "\n", xdata, "\n", ydata)
    line.set_data(xdata, ydata)
    return line,


def run(data):
    # update the data
    t, y = data
    print("run", type(data), data)
    if(y is None):
        return
    xdata.append(t)
    ydata.append(y)
    # print("run ydata", ydata)

    # increase canvas size, if needed
    xmin, xmax = ax.get_xlim()
    if t >= xmax:
        ax.set_xlim(xmin, 2*xmax)
        ax.figure.canvas.draw()
    ymin, ymax = ax.get_xlim()
    if y >= ymax:
        ax.set_ylim(ymin, y+1)
        ax.figure.canvas.draw()

    print("run set_data", xdata, ydata)
    line.set_data(xdata, ydata)

    return line,


xdata, ydata = [], []
for doc in collection.find({"type": "inverse"}, sort=[("_id", pymongo.ASCENDING)]):
    ydata.append(doc["APHu"])
xdata = list(range(len(ydata)))
print(xdata, "\n", ydata)

fig, ax = plt.subplots()
line, = ax.plot(xdata, ydata, lw=2)
ax.grid()

ani = animation.FuncAnimation(fig, run, data_gen(len(xdata)), blit=False, interval=1000,
                              repeat=False, init_func=init)
plt.show()
