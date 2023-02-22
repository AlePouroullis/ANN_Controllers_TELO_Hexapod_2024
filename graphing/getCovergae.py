import pandas as pd
import numpy as np

## Load all 5k Map Fitness in
dfReadIn = pd.read_csv(
    rf'C:\Users\micha\PycharmProjects\HonoursProject\mapElitesOutput\NEAT\0_5000\log.dat',
    header=None, sep='\s', engine='python')
dfReadIn5 = pd.read_csv(
    rf'C:\Users\micha\PycharmProjects\HonoursProject\mapElitesOutput\NEAT\0_5000\log.dat',
    header=None, sep='\s', engine='python')
dfReadIn10 = pd.read_csv(
    rf'C:\Users\micha\PycharmProjects\HonoursProject\mapElitesOutput\NEAT\0_10000\log.dat',
    header=None, sep='\s', engine='python')
dfReadIn20 = pd.read_csv(
    rf'C:\Users\micha\PycharmProjects\HonoursProject\mapElitesOutput\NEAT\0_20000\log.dat',
    header=None, sep='\s', engine='python')
dfReadIn40 = pd.read_csv(
    rf'C:\Users\micha\PycharmProjects\HonoursProject\mapElitesOutput\NEAT\0_40000\log.dat',
    header=None, sep='\s', engine='python')
dfReadIn80 = pd.read_csv(
    rf'C:\Users\micha\PycharmProjects\HonoursProject\mapElitesOutput\NEAT\0_80000\log.dat',
    header=None, sep='\s', engine='python')

dfMainNEAT = pd.DataFrame(columns=['5k', '10k', '20k', '40k', '80k'])
coverage5 = np.array([dfReadIn5[dfReadIn.columns[1]].tail(1)])
coverage10 = np.array([dfReadIn10[dfReadIn.columns[1]].tail(1)])
coverage20 = np.array([dfReadIn20[dfReadIn.columns[1]].tail(1)])
coverage40 = np.array([dfReadIn40[dfReadIn.columns[1]].tail(1)])
coverage80 = np.array([dfReadIn80[dfReadIn.columns[1]].tail(1)])

for i in range(1, 10):
    print(i)
    dfReadIn5 = pd.read_csv(rf'C:\Users\micha\PycharmProjects\HonoursProject\mapElitesOutput\NEAT\{i}_5000\log.dat', header=None, sep='\s', engine='python')
    coverage5 = np.append(coverage5,([dfReadIn5[dfReadIn.columns[1]].tail(1)]))

    dfReadIn10 = pd.read_csv(rf'C:\Users\micha\PycharmProjects\HonoursProject\mapElitesOutput\NEAT\{i}_10000\log.dat', header=None, sep='\s', engine='python')
    coverage10 = np.append(coverage10,([dfReadIn10[dfReadIn.columns[1]].tail(1)]))

    if i == 9:
        coverage20 = np.append(coverage20, 19999)
    else:
        dfReadIn20 = pd.read_csv(rf'C:\Users\micha\PycharmProjects\HonoursProject\mapElitesOutput\NEAT\{i}_20000\log.dat', header=None, sep='\s', engine='python')
        coverage20 = np.append(coverage20,([dfReadIn20[dfReadIn.columns[1]].tail(1)]))

    if (i == 3):
        coverage40 = np.append(coverage40, 39934)
    else:
        dfReadIn40 = pd.read_csv(rf'C:\Users\micha\PycharmProjects\HonoursProject\mapElitesOutput\NEAT\{i}_40000\log.dat', header=None, sep='\s', engine='python')
        coverage40 = np.append(coverage40,([dfReadIn40[dfReadIn.columns[1]].tail(1)]))

    dfReadIn80 = pd.read_csv(rf'C:\Users\micha\PycharmProjects\HonoursProject\mapElitesOutput\NEAT\{i}_80000\log.dat', header=None, sep='\s', engine='python')
    coverage80 = np.append(coverage80,([dfReadIn80[dfReadIn.columns[1]].tail(1)]))

coverage5 = coverage5/5000
coverage10 = coverage10/10000
coverage20 = coverage20/20000
coverage40 = coverage40/40000
coverage80 = coverage80/80000

dfCoverageNEAT = pd.DataFrame(data=[coverage5, coverage10, coverage20, coverage40, coverage80])
dfCoverageNEAT = dfCoverageNEAT.transpose()
dfCoverageNEAT.columns = ['5k', '10k', '20k', '40k', '80k']
dfCoverageNEAT.to_csv("NEATCoverage")
print("hi")



## Load all 5k Map Fitness in
dfReadIn = pd.read_csv(
    rf'C:\Users\micha\PycharmProjects\HonoursProject\mapElitesOutput\HyperNEAT\0_5000\log.dat',
    header=None, sep='\s', engine='python')
dfReadIn5 = pd.read_csv(
    rf'C:\Users\micha\PycharmProjects\HonoursProject\mapElitesOutput\HyperNEAT\0_5000\log.dat',
    header=None, sep='\s', engine='python')
dfReadIn10 = pd.read_csv(
    rf'C:\Users\micha\PycharmProjects\HonoursProject\mapElitesOutput\HyperNEAT\0_10000\log.dat',
    header=None, sep='\s', engine='python')
dfReadIn20 = pd.read_csv(
    rf'C:\Users\micha\PycharmProjects\HonoursProject\mapElitesOutput\HyperNEAT\0_20000\log.dat',
    header=None, sep='\s', engine='python')
dfReadIn40 = pd.read_csv(
    rf'C:\Users\micha\PycharmProjects\HonoursProject\mapElitesOutput\HyperNEAT\0_40000\log.dat',
    header=None, sep='\s', engine='python')
dfReadIn80 = pd.read_csv(
    rf'C:\Users\micha\PycharmProjects\HonoursProject\mapElitesOutput\HyperNEAT\0_80000\log.dat',
    header=None, sep='\s', engine='python')

dfMainHyperNEAT = pd.DataFrame(columns=['5k', '10k', '20k', '40k', '80k'])
coverage5 = np.array([dfReadIn5[dfReadIn.columns[1]].tail(1)])
coverage10 = np.array([dfReadIn10[dfReadIn.columns[1]].tail(1)])
coverage20 = np.array([dfReadIn20[dfReadIn.columns[1]].tail(1)])
coverage40 = np.array([dfReadIn40[dfReadIn.columns[1]].tail(1)])
coverage80 = np.array([dfReadIn80[dfReadIn.columns[1]].tail(1)])

for i in range(1, 10):
    print(i)
    dfReadIn5 = pd.read_csv(rf'C:\Users\micha\PycharmProjects\HonoursProject\mapElitesOutput\HyperNEAT\{i}_5000\log.dat', header=None, sep='\s', engine='python')
    coverage5 = np.append(coverage5,([dfReadIn5[dfReadIn.columns[1]].tail(1)]))

    dfReadIn10 = pd.read_csv(rf'C:\Users\micha\PycharmProjects\HonoursProject\mapElitesOutput\HyperNEAT\{i}_10000\log.dat', header=None, sep='\s', engine='python')
    coverage10 = np.append(coverage10,([dfReadIn10[dfReadIn.columns[1]].tail(1)]))

    if i == 9:
        coverage20 = np.append(coverage20, 19999)
    else:
        dfReadIn20 = pd.read_csv(rf'C:\Users\micha\PycharmProjects\HonoursProject\mapElitesOutput\HyperNEAT\{i}_20000\log.dat', header=None, sep='\s', engine='python')
        coverage20 = np.append(coverage20,([dfReadIn20[dfReadIn.columns[1]].tail(1)]))

    if (i == 3):
        coverage40 = np.append(coverage40, 39934)
    else:
        dfReadIn40 = pd.read_csv(rf'C:\Users\micha\PycharmProjects\HonoursProject\mapElitesOutput\HyperNEAT\{i}_40000\log.dat', header=None, sep='\s', engine='python')
        coverage40 = np.append(coverage40,([dfReadIn40[dfReadIn.columns[1]].tail(1)]))

    dfReadIn80 = pd.read_csv(rf'C:\Users\micha\PycharmProjects\HonoursProject\mapElitesOutput\HyperNEAT\{i}_80000\log.dat', header=None, sep='\s', engine='python')
    coverage80 = np.append(coverage80,([dfReadIn80[dfReadIn.columns[1]].tail(1)]))

coverage5 = coverage5/5000
coverage10 = coverage10/10000
coverage20 = coverage20/20000
coverage40 = coverage40/40000
coverage80 = coverage80/80000

dfCoverageHyperNEAT = pd.DataFrame(data=[coverage5, coverage10, coverage20, coverage40, coverage80])
dfCoverageHyperNEAT = dfCoverageHyperNEAT.transpose()
dfCoverageHyperNEAT.columns = ['5k', '10k', '20k', '40k', '80k']
dfCoverageHyperNEAT.to_csv("HyperNEATCoverage")
print("hi")
