#-----------------------------------------------------------------------------------------------------------------------------------
__author__ = "Christian Simonis"
__copyright__ = "Copyright 2023"
__version__ = "1.0"
__maintainer__ = "Christian Simonis"
__email__ = "christian.Simonis.1989@gmail.com"
__status__ = "work in progress"
#-----------------------------------------------------------------------------------------------------------------------------------
import pickle
import gpflow
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import MinMaxScaler
from pathlib import Path

#-----------------------------------------------------------------------------------------------------------------------------------
bdir = Path(__file__).resolve(strict=True).parent
#-----------------------------------------------------------------------------------------------------------------------------------

with open(f"{bdir}/model_pls.pkl", "rb") as f:
    loaded_mdl = pickle.load(f)
    scaler = pickle.load(f)
    pls = pickle.load(f)
    X_test = pickle.load(f)

def execute_mdl(data):

    # Scale Data
    data_scaled = scaler.transform(data)

    # Apply PLS transformation
    data_pls = pls.transform(data_scaled)

    # Execute probabilistic model
    mean, var = loaded_mdl.predict_f(data_pls)
    return mean, var