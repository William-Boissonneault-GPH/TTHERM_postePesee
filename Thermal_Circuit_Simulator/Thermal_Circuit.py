import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from pathlib import Path
import datetime as dt

class ThermalCircuit:
    def __init__(self):
        self.capacitances = []
        self.resistances = {}
        self.n_nodes = 0

    # Add a thermal node with its capacitance
    def add_node(self, C):
        self.capacitances.append(C)
        self.n_nodes += 1
        return self.n_nodes - 1       # return index of the node
    
    def set_outdoor_node(self, time, temperature):
        self.capacitances.append(np.inf)  # Outdoor node with infinite capacitance
        self.n_nodes += 1
        self.outdoor_time = time
        self.outdoor_temperature = temperature
        self.outdoor_node_index = self.n_nodes - 1
        return self.n_nodes - 1
        
    # Set resistance between node i and j
    def set_resistance(self, i, j, R):
        if i == j:
            raise ValueError("No self-resistance allowed.")
        self.resistances[(i, j)] = R
        self.resistances[(j, i)] = R  # symmetric

    # # TODO
    # def set_heated_node(self, node_index, q, condition=lambda t: True):
    #     if not hasattr(self, 'heated_nodes'):
    #         self.heated_nodes = {}
    #     self.heated_nodes[node_index] = (q, condition)

    # Build G (conductance matrix) and C (capacitance diag)
    def build_matrices(self):
        N = self.n_nodes 
        C = np.diag(self.capacitances)
        G = np.zeros((N, N))

        for Resistance_Connexions in self.resistances.items():
            (i, j), R = Resistance_Connexions
            G[i, j] = -1.0 / R

        # Diagonal elements = sum of outgoing conductances
        for i in range(N):
            G[i, i] = -np.sum(G[i, :])  # row must sum to zero

        return C, G

    # Simulate with explicit Euler
    def simulate(self, T0, dt, total_time):
        C, G = self.build_matrices()
        N_steps = int(total_time / dt)

        T = np.zeros((N_steps + 1, self.n_nodes))
        T[0, :] = T0

        # Precompute C^{-1}
        C_inv = np.linalg.inv(C)

        # Time vector
        time = np.linspace(0, total_time, N_steps + 1)

        # Precompute closest outdoor temperatures if available
        if hasattr(self, 'outdoor_time') and hasattr(self, 'outdoor_temperature'):
            outdoor_index = self.outdoor_node_index
            outdoor_temp_full = np.zeros(N_steps + 1)
            for n, t in enumerate(time):
                # find index of the closest outdoor_time
                idx = np.abs(np.array(self.outdoor_time) - t).argmin()
                outdoor_temp_full[n] = self.outdoor_temperature[idx]
        else:
            outdoor_temp_full = None

        # Simulation loop
        print('Beginning simulation...')
        for n in range(N_steps):
            dT = C_inv @ (-G @ T[n, :])
            T[n+1, :] = T[n, :] + dt * dT
            # Update outdoor node
            if outdoor_temp_full is not None:
                T[n+1, outdoor_index] = outdoor_temp_full[n+1]

        # Return both T and time vector
        return T, time
    
    def show_temperature_graph(self, T, time, dates_axis=None):
        plt.figure(figsize=(8, 5))
        for i in range(self.n_nodes):
            plt.plot(time, T[:, i], label=f"Node {i}")
        if dates_axis is not None:
            # Get 15 evenly spaced indices including first and last
            indices = np.linspace(0, len(dates_axis) - 1, 15, dtype=int)
            plt.xticks(ticks=time[indices], labels=dates_axis[indices], rotation=45)
            plt.xlabel("Time (dates)")
        else:
            plt.xlabel("Time (seconds)")
        plt.ylabel("Temperature (°C)")
        plt.title("Temperature Evolution of Thermal Nodes")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

def get_experimental_data():
    script_directory = os.path.dirname(os.path.abspath(__file__))
    irl_data_path = os.path.join(script_directory, "doi-10.5683-sp3-iaas16", "Dataset of weighing station temperature measurements.csv")
    measurement_data_df = pd.read_csv(irl_data_path, sep=';')
    outdoor_temperature_data = measurement_data_df['Outdoor temperature [deg. C]'].to_list()
    simulation_time_in_dates = pd.to_datetime(measurement_data_df['Time'].to_list())
    simulation_time = (simulation_time_in_dates - simulation_time_in_dates[0]).total_seconds()  # Convert to absolute seconds
    return outdoor_temperature_data, simulation_time_in_dates, simulation_time
    
    
# Exemple de Circuit
if __name__ == "__main__":
    tc = ThermalCircuit()

    # Get experimental data from CSV
    outdoor_temperature_data, simulation_time_in_dates, simulation_time = get_experimental_data

    # Add les nodes(avec les Capacitances en [J/K])
    n0 = tc.add_node(C=34400)
    n1 = tc.add_node(C=4.2*10**6)
    n2 = tc.add_node(C=4.2*10**6)
    n3 = tc.set_outdoor_node(simulation_time, outdoor_temperature_data)

    # Set les resistances entre les nodes(en [K/W])
    tc.set_resistance(n0, n1, R=0.125)
    tc.set_resistance(n0, n2, R=0.125)
    tc.set_resistance(n0, n3, R=0.055)

    # Heated nodes Get q injected if conditions are True
    # tc.set_heated_node(n0, q, condition=lambda t: (< 5.0)) TODO

    # Températures initiales # TODO Set from simulation data
    T0 = np.zeros(tc.n_nodes)
    T0[n0] = 100.0  # Node 0 starts at 100°C
    T0[n1] = 20.0   # Node 1 starts at 20°C
    T0[n2] = 50.0   # Node 2 starts at 50°C
    T0[n3] = outdoor_temperature_data[0]  # Outdoor node initial temperature

    # Simulation(en secondes)
    dt = 120 # 2 min en secondes
    total_time = 108094*60 # en secondes
    simulation_temperature_matrix, simulation_time = tc.simulate(T0, dt, total_time)

    # Graphe de la simu
    tc.show_temperature_graph(simulation_temperature_matrix, simulation_time, dates_axis = simulation_time_in_dates)
    # tc.compare_with_experimental_data(simulation_temperature_matrix, simulation_time, ...)  # TODO
    # TODO Sortir la diference de Tair mais Tair data doit etre pondere avec le bon volume