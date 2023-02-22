from hexapod.controllers.NEATController import Controller, reshape, stationary
from hexapod.simulator import Simulator
from adapt.MBOA import MBOA
import numpy as np
import pickle
import neat
import sys

"""
A script used to run adaptation test for all damage scenario's 

The script takes two command line arguments:
1) 0 or 1 indicate whether we are running the test for 20k or 40k maps.
2) An integer [0, 4] indicating which damage scenario we are testing.
"""

# Configure ann using config file
config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                     neat.DefaultSpeciesSet, neat.DefaultStagnation,
                     'NEATHex/config-feedforward')

maps5File = ["mapElitesOutput/NEAT/0_5000archive/archive10002200.dat",
             "mapElitesOutput/NEAT/1_5000archive/archive10002200.dat",
             "mapElitesOutput/NEAT/2_5000archive/archive10002200.dat",
             "mapElitesOutput/NEAT/3_5000archive/archive10002200.dat",
             "mapElitesOutput/NEAT/4_5000archive/archive10002200.dat",
             "mapElitesOutput/NEAT/5_5000archive/archive10002200.dat",
             "mapElitesOutput/NEAT/6_5000archive/archive10002200.dat",
             "mapElitesOutput/NEAT/7_5000archive/archive10002200.dat",
             "mapElitesOutput/NEAT/8_5000archive/archive10002200.dat",
             "mapElitesOutput/NEAT/9_5000archive/archive10002200.dat"]

maps5Genome = ["mapElitesOutput/NEAT/0_5000archive/archive_genome10002200.pkl",
               "mapElitesOutput/NEAT/1_5000archive/archive_genome10002200.pkl",
               "mapElitesOutput/NEAT/2_5000archive/archive_genome10002200.pkl",
               "mapElitesOutput/NEAT/3_5000archive/archive_genome10002200.pkl",
               "mapElitesOutput/NEAT/4_5000archive/archive_genome10002200.pkl",
               "mapElitesOutput/NEAT/5_5000archive/archive_genome10002200.pkl",
               "mapElitesOutput/NEAT/6_5000archive/archive_genome10002200.pkl",
               "mapElitesOutput/NEAT/7_5000archive/archive_genome10002200.pkl",
               "mapElitesOutput/NEAT/8_5000archive/archive_genome10002200.pkl",
               "mapElitesOutput/NEAT/9_5000archive/archive_genome10002200.pkl"]

maps10File = ["mapElitesOutput/NEAT/0_10000archive/archive10002250.dat",
              "mapElitesOutput/NEAT/1_10000archive/archive10002250.dat",
              "mapElitesOutput/NEAT/2_10000archive/archive10002250.dat",
              "mapElitesOutput/NEAT/3_10000archive/archive10002250.dat",
              "mapElitesOutput/NEAT/4_10000archive/archive10002250.dat",
              "mapElitesOutput/NEAT/5_10000archive/archive10002250.dat",
              "mapElitesOutput/NEAT/6_10000archive/archive10002250.dat",
              "mapElitesOutput/NEAT/7_10000archive/archive10002250.dat",
              "mapElitesOutput/NEAT/8_10000archive/archive10002250.dat",
              "mapElitesOutput/NEAT/9_10000archive/archive10002250.dat"]

maps10Genome = ["mapElitesOutput/NEAT/0_10000archive/archive_genome10002250.pkl",
              "mapElitesOutput/NEAT/1_10000archive/archive_genome10002250.pkl",
              "mapElitesOutput/NEAT/2_10000archive/archive_genome10002250.pkl",
              "mapElitesOutput/NEAT/3_10000archive/archive_genome10002250.pkl",
              "mapElitesOutput/NEAT/4_10000archive/archive_genome10002250.pkl",
              "mapElitesOutput/NEAT/5_10000archive/archive_genome10002250.pkl",
              "mapElitesOutput/NEAT/6_10000archive/archive_genome10002250.pkl",
              "mapElitesOutput/NEAT/7_10000archive/archive_genome10002250.pkl",
              "mapElitesOutput/NEAT/8_10000archive/archive_genome10002250.pkl",
              "mapElitesOutput/NEAT/9_10000archive/archive_genome10002250.pkl"]
# Maps to be tested
maps20File = ["mapElitesOutput/NEAT/0_20000archive/archive10002350.dat",
              "mapElitesOutput/NEAT/1_20000archive/archive10002350.dat",
              "mapElitesOutput/NEAT/2_20000archive/archive10002350.dat",
              "mapElitesOutput/NEAT/3_20000archive/archive10002350.dat",
              "mapElitesOutput/NEAT/4_20000archive/archive10002350.dat",
              "mapElitesOutput/NEAT/5_20000archive/archive10002350.dat",
              "mapElitesOutput/NEAT/6_20000archive/archive10002350.dat",
              "mapElitesOutput/NEAT/7_20000archive/archive10002350.dat",
              "mapElitesOutput/NEAT/8_20000archive/archive10002350.dat"
              ]

maps40File = ["mapElitesOutput/NEAT/0_40000archive/archive10000160.dat",
              "mapElitesOutput/NEAT/1_40000archive/archive10000160.dat",
              "mapElitesOutput/NEAT/2_40000archive/archive10000160.dat",
              "mapElitesOutput/NEAT/3_40000archive/archive10000160.dat",
              "mapElitesOutput/NEAT/4_40000archive/archive10000160.dat",
              "mapElitesOutput/NEAT/5_40000archive/archive10000160.dat",
              "mapElitesOutput/NEAT/6_40000archive/archive10000160.dat",
              "mapElitesOutput/NEAT/7_40000archive/archive10000160.dat",
              "mapElitesOutput/NEAT/8_40000archive/archive10000160.dat",
              "mapElitesOutput/NEAT/9_40000archive/archive10000160.dat",
              ]

maps20Genome = ["mapElitesOutput/NEAT/0_20000archive/archive_genome10002350.pkl",
                "mapElitesOutput/NEAT/1_20000archive/archive_genome10002350.pkl",
                "mapElitesOutput/NEAT/2_20000archive/archive_genome10002350.pkl",
                "mapElitesOutput/NEAT/3_20000archive/archive_genome10002350.pkl",
                "mapElitesOutput/NEAT/4_20000archive/archive_genome10002350.pkl",
                "mapElitesOutput/NEAT/5_20000archive/archive_genome10002350.pkl",
                "mapElitesOutput/NEAT/6_20000archive/archive_genome10002350.pkl",
                "mapElitesOutput/NEAT/7_20000archive/archive_genome10002350.pkl",
                "mapElitesOutput/NEAT/8_20000archive/archive_genome10002350.pkl"
                ]

maps40Genome = ["mapElitesOutput/NEAT/0_40000archive/archive_genome10000160.pkl",
                "mapElitesOutput/NEAT/1_40000archive/archive_genome10000160.pkl",
                "mapElitesOutput/NEAT/2_40000archive/archive_genome10000160.pkl",
                "mapElitesOutput/NEAT/3_40000archive/archive_genome10000160.pkl",
                "mapElitesOutput/NEAT/4_40000archive/archive_genome10000160.pkl",
                "mapElitesOutput/NEAT/5_40000archive/archive_genome10000160.pkl",
                "mapElitesOutput/NEAT/6_40000archive/archive_genome10000160.pkl",
                "mapElitesOutput/NEAT/7_40000archive/archive_genome10000160.pkl",
                "mapElitesOutput/NEAT/8_40000archive/archive_genome10000160.pkl",
                "mapElitesOutput/NEAT/9_40000archive/archive_genome10000160.pkl"
                ]

maps80File = ["mapElitesOutput/NEAT/0_80000archive/archive10000560.dat",
              "mapElitesOutput/NEAT/1_80000archive/archive10000560.dat",
              "mapElitesOutput/NEAT/2_80000archive/archive10000560.dat",
              "mapElitesOutput/NEAT/3_80000archive/archive10000560.dat",
              "mapElitesOutput/NEAT/4_80000archive/archive10000560.dat",
              "mapElitesOutput/NEAT/5_80000archive/archive10000560.dat",
              "mapElitesOutput/NEAT/6_80000archive/archive10000560.dat",
              "mapElitesOutput/NEAT/7_80000archive/archive10000560.dat",
              "mapElitesOutput/NEAT/8_80000archive/archive10000560.dat",
              "mapElitesOutput/NEAT/9_80000archive/archive10000560.dat"]

maps80Genome = ["mapElitesOutput/NEAT/0_80000archive/archive_genome10000560.pkl",
                "mapElitesOutput/NEAT/1_80000archive/archive_genome10000560.pkl",
                "mapElitesOutput/NEAT/2_80000archive/archive_genome10000560.pkl",
                "mapElitesOutput/NEAT/3_80000archive/archive_genome10000560.pkl",
                "mapElitesOutput/NEAT/4_80000archive/archive_genome10000560.pkl",
                "mapElitesOutput/NEAT/5_80000archive/archive_genome10000560.pkl",
                "mapElitesOutput/NEAT/6_80000archive/archive_genome10000560.pkl",
                "mapElitesOutput/NEAT/7_80000archive/archive_genome10000560.pkl",
                "mapElitesOutput/NEAT/8_80000archive/archive_genome10000560.pkl",
                "mapElitesOutput/NEAT/9_80000archive/archive_genome10000560.pkl",]

# Define which maps and failure scenario we are testing
mapType = int(sys.argv[1])
failure_scenario = int(sys.argv[2])
mapsFile = ''
if mapType == 10000:
    niches = 10000
    mapsFile = maps10File
    mapsGenome = maps10Genome
elif mapType == 20000:
    niches = 20000
    mapsFile = maps20File
    mapsGenome = maps20Genome
elif mapType == 5000:
    niches = 5000
    mapsFile = maps5File
    mapsGenome = maps5Genome
elif mapType == 80000:
    niches = 80000
    mapsFile = maps80File
    mapsGenome = maps80Genome
else:
    niches = 40000
    mapsFile = maps40File
    mapsGenome = maps40Genome
map_count = 10

# The different damage configurations
S0 = [[]]
S1 = [[1], [2], [3], [4], [5], [6]]
S2 = [[1, 4], [2, 5], [3, 6]]
S3 = [[1, 3], [2, 4], [3, 5], [4, 6], [5, 1], [6, 2]]
S4 = [[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 1]]

scenarios = [S0, S1, S2, S3, S4]
failures = scenarios[failure_scenario]

# Set up empty numpy arrays for the relevant output
num_its = np.zeros((len(failures), map_count))
best_indexes = np.zeros((len(failures), map_count))
best_perfs = np.zeros((len(failures), map_count))

# Iterate through different failures and maps for each scenario
for failure_index, failed_legs in enumerate(failures):
    print("Failed legs:", failed_legs)
    for map_num in range(0, map_count):
        print("Testing map:", map_num)


        # NEAT evaluation function
        def evaluate_gait(x, duration=5):
            net = neat.nn.FeedForwardNetwork.create(x, config)
            # Reset net

            leg_params = np.array(stationary).reshape(6, 5)
            # Set up controller
            try:
                controller = Controller(leg_params, body_height=0.15, velocity=0.5, period=1.0, crab_angle=-np.pi / 6,
                                        ann=net)
            except:
                return 0, np.zeros(6)
            # Initialise Simulator
            simulator = Simulator(controller=controller, visualiser=False, collision_fatal=False,
                                  failed_legs=failed_legs)
            # Step in simulator
            contact_sequence = np.full((6, 0), False)
            for t in np.arange(0, duration, step=simulator.dt):
                try:
                    simulator.step()
                except RuntimeError as collision:
                    fitness = 0, np.zeros(6)
                contact_sequence = np.append(contact_sequence, simulator.supporting_legs().reshape(-1, 1), axis=1)
            fitness = simulator.base_pos()[0]  # distance travelled along x axis
            descriptor = np.nan_to_num(np.sum(contact_sequence, axis=1) / np.size(contact_sequence, axis=1), nan=0.0,
                                       posinf=0.0, neginf=0.0)
            # Terminate Simulator
            simulator.terminate()
            # Assign fitness to genome
            x.fitness = fitness
            return fitness


        # Determine which map to perform Optimisaiton on and open map and pickle file containing the genomes
        mapGenome = mapsGenome[map_num]
        mapFile = mapsFile[map_num]
        with open(mapGenome, 'rb') as f:
            genomes = pickle.load(f)
        num_it, best_index, best_perf, new_map = MBOA(mapFile, genomes, f"./centroids/centroids_{niches}_6.dat",
                                                      evaluate_gait, max_iter=50, print_output=False)
        # Update relevant output files
        print(num_it)
        num_its[failure_index, map_num - 1] = num_it
        best_indexes[failure_index, map_num - 1] = best_index
        best_perfs[failure_index, map_num - 1] = best_perf


# Save output
np.savetxt(f"./mapElitesOutput/NEATSim/{niches}_niches/trials_{failure_scenario}.dat", num_its, '%d')
np.savetxt(f"./mapElitesOutput/NEATSim/{niches}_niches/perfs_{failure_scenario}.dat", best_perfs)
np.savetxt(f"./mapElitesOutput/NEATSim/{niches}_niches/indexes_{failure_scenario}.dat", best_indexes)
