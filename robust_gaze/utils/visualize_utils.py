import plotly.express as px
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
try :
    import plotly.express as px
except:
    print("plotly not installed, some functions will not work")


def plot_3d_vertices(vertices: list[np.ndarray],save_file):
    for i,v in enumerate(vertices):
        vertices[i] = np.hstack((v,i*np.ones((len(v),1))))
    
    X = np.concatenate(vertices,axis = 1)
    df = pd.DataFrame(X,columns = ['x','y','z','objects'])

    fig = px.scatter_3d(df, x='x', y='y', z='z',
                        color='objects',size_max=0.1)
    plt.savefig(save_file)