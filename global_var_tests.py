

point = 1

def f(x):
    global point
    print(point)
    point = 2

f(2)
print(point)
