import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pymongo
from pymongo import MongoClient

client = MongoClient("mongodb://localhost:27017")
print(client)
db = client.simulation
print(db)
data_collection = db.data
print(data_collection)
gaussian_collection = db.gaussian
print(gaussian_collection)
generated_collection = db.generated
print(generated_collection)


def data_gen(t=0):
    # loop for each new data point
    print("data_gen", t)
    while True:
        doc = generated_collection.find_one({"type": "inverse"}, sort=[("_id", pymongo.ASCENDING)], skip=t)
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


real_data_y = []
for doc in data_collection.find({}, sort=[("_id", pymongo.ASCENDING)]):
    real_data_y.append(doc["APHu"])
real_data_x = list(range(len(real_data_y)))

gaussian_data_y = []
for doc in gaussian_collection.find({"type":"inverse"}, sort=[("_id", pymongo.ASCENDING)], limit=len(real_data_y)*2):
    gaussian_data_y.append(doc["APHu"])
gaussian_data_x = list(range(len(gaussian_data_y)))

xdata, ydata = [], []
for doc in generated_collection.find({"type": "inverse"}, sort=[("_id", pymongo.ASCENDING)]):
    ydata.append(doc["APHu"])
xdata = list(range(len(ydata)))
print(xdata, "\n", ydata)

fig, ax = plt.subplots()
ax.plot(real_data_x, real_data_y, label="original")
ax.plot(gaussian_data_x, gaussian_data_y, label="gaussian")


line, = ax.plot(xdata, ydata, label="predicted")
# ax.plot([0,1,2,3,4], [33,23,11,22,44])
# ax.plot([10,11,12,13,14], [101,120,102,210,500])

ax.axhline(np.max(gaussian_data_y), color="gray", linewidth=1)
ax.axhline(np.min(gaussian_data_y), color="gray", linewidth=1)
ax.axhline(np.mean(gaussian_data_y), color="gray", linewidth=1)
ax.axhline(np.mean(gaussian_data_y) + 2 * np.std(gaussian_data_y), color="gray", linewidth=1)
ax.axhline(np.mean(gaussian_data_y) + 3 * np.std(gaussian_data_y), color="gray", linewidth=1)
ax.axhline(np.mean(gaussian_data_y) - 2 * np.std(gaussian_data_y), color="gray", linewidth=1)
ax.axhline(np.mean(gaussian_data_y) - 3 * np.std(gaussian_data_y), color="gray", linewidth=1)
title = "dynamic plot"
plt.title(title)
ax.legend()
ax.grid()

plt.show()

exit()

ani = animation.FuncAnimation(fig, run, data_gen(len(xdata)), blit=False, interval=1000,
                              repeat=False, init_func=init)
plt.show()
