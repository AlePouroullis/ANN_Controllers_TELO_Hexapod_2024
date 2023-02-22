import numpy as np
import pandas as pd

# sim_data_NEAT10 = []
# Load in all Data
# for scenario in range(5):
#     data_NEAT10 = list(np.loadtxt(
#         rf"C:\Users\micha\PycharmProjects\HonoursProject\mapElitesOutput\NEATSim\10000_niches\perfs_{scenario}.dat").flatten())
#     sim_data_NEAT10.append(data_NEAT10)
#
# dfNEAT10 = pd.DataFrame(sim_data_NEAT10)
# dfNEAT10 = dfNEAT10.transpose()
# dfNEAT10.to_csv('NEAT10perfs.csv')


sim_data_HyperNEAT5 = []
for scenario in range(5):
    data_HyperNEAT5 = list(np.loadtxt(
        rf"C:\Users\micha\PycharmProjects\HonoursProject\mapElitesOutput\HyperNEATSim\5000_niches\perfs_{scenario}.dat").flatten())
    sim_data_HyperNEAT5.append(data_HyperNEAT5)

dfHyperNEAT5 = pd.DataFrame(sim_data_HyperNEAT5)
dfHyperNEAT5 = dfHyperNEAT5.transpose()
dfHyperNEAT5.to_csv('HyperNEAT5perfs.csv')
