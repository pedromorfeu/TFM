import numpy as np
import matplotlib.pylab as plt


def generate(fR=0.05):
    x = np.linspace(-np.pi, 10*np.pi, 3000)
    print(len(x))
    sin0=np.sin(x)
    std0=np.std(sin0)
    noise=np.random.normal(0,std0*fR,len(x))
    sin=np.add(sin0,noise)


    sin0_3=3*np.sin(x)
    std=np.std(sin0_3)
    noise=np.random.normal(0,std*fR,len(x))
    sin_3=np.add(sin0_3,noise)

    sin_sin_3=np.add(sin,sin_3)

    print(np.corrcoef(sin,sin_sin_3))

    print(x)

    plt.plot(x)
    plt.plot(sin_sin_3)
    plt.plot(x, sin_sin_3)
    plt.xlabel('Angle [rad]')

    plt.ylabel('sin(x)')
    plt.axis('tight')
    # plt.show()


generate(0.09)
plt.show()