import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pymongo import MongoClient

client = MongoClient("mongodb://localhost:27017")
print(client)
db = client.simulation
print(db)
collection = db.generated
print(collection)


def data_gen(t=0):
    # loop for each new data point
    while True:
        doc = collection.find_one(skip=t)
        if doc is not None:
            t += 1
            yield t, doc["number"]
        else:
            yield t, 0

def init():
    ax.set_ylim(-3, 3)
    ax.set_xlim(0, len(xdata))
    # del xdata[:]
    # del ydata[:]
    line.set_data(xdata, ydata)
    return line,


fig, ax = plt.subplots()
line, = ax.plot([], [], lw=2)
ax.grid()

xdata, ydata = [], []
for doc in collection.find():
    ydata.append(doc["number"])
xdata = list(range(len(ydata)))
print(len(ydata))
print(len(xdata))


def run(data):
    # update the data
    t, y = data
    # print("run", data)
    xdata.append(t)
    ydata.append(y)
    xmin, xmax = ax.get_xlim()

    if t >= xmax:
        ax.set_xlim(xmin, 2*xmax)
        ax.figure.canvas.draw()
    line.set_data(xdata, ydata)

    return line,

ani = animation.FuncAnimation(fig, run, data_gen(len(xdata)), blit=False, interval=500,
                              repeat=False, init_func=init)
plt.show()
