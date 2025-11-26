import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from pathlib import Path
import datetime as dt

def get_experimental_data():
    script_directory = os.path.dirname(os.path.abspath(__file__))
    irl_data_path = os.path.join(script_directory, "doi-10.5683-sp3-iaas16", "Dataset of weighing station temperature measurements.csv")
    measurement_data_df = pd.read_csv(irl_data_path, sep=';')
    outdoor_temperature_data = measurement_data_df['Outdoor temperature [deg. C]'].to_list()
    simulation_time_in_dates = pd.to_datetime(measurement_data_df['Time'].to_list())
    simulation_time = (simulation_time_in_dates - simulation_time_in_dates[0]).total_seconds()  # Convert to absolute seconds

    low_cols = [col for col in measurement_data_df.columns if "T[degC]-Low" in col]
    mid_cols = [col for col in measurement_data_df.columns if "T[degC]-Mid" in col]
    top_cols = [col for col in measurement_data_df.columns if "T[degC]-Top" in col]

    measurement_data_df["Low_avg"] = measurement_data_df[low_cols].mean(axis=1)
    measurement_data_df["Mid_avg"] = measurement_data_df[mid_cols].mean(axis=1)
    measurement_data_df["Top_avg"] = measurement_data_df[top_cols].mean(axis=1)

    weighted_total_average = (
        0.35 * measurement_data_df["Low_avg"] +
        0.23 * measurement_data_df["Mid_avg"] +
        0.411 * measurement_data_df["Top_avg"]
    )
    measurement_data_df["Total_Average"] = weighted_total_average

    return outdoor_temperature_data, simulation_time_in_dates, simulation_time, measurement_data_df

class ThermalCircuit:
    def __init__(self):
        self.capacitances = []
        self.resistances = {}
        self.heated_nodes = []
        self.outdoor_node_index = []
        self.node_names = []
        self.n_nodes = 0
        out_temp, simu_time_date, simu_time_sec, data_df = get_experimental_data()
        self.outdoor_temperature_data = out_temp
        self.simulation_time_in_dates = simu_time_date
        self.simulation_time_in_seconds = simu_time_sec
        self.measurement_data_df = data_df

    # Add a thermal node with its capacitance
    def add_node(self, C, name=None):
        """
        Add a thermal node with capacitance C [J/K].
        Si tu lui donne un nom, il va apparaitre dans le graphe.
        """
        self.capacitances.append(C)
        self.n_nodes += 1
        self.node_names.append(name)
        return self.n_nodes - 1       # return index of the node
    
    def add_outdoor_node(self, name=None):
        """
        Add an outdoor node with infinite capacitance.
        Si tu lui donne un nom, il va apparaitre dans le graphe.
        """
        self.capacitances.append(np.inf)  # Outdoor node with infinite capacitance
        self.n_nodes += 1
        self.outdoor_node_index.append(self.n_nodes - 1)
        self.node_names.append(name)
        return self.n_nodes - 1
    
    def add_ground_node(self, profondeur=1, name=None):
        """
        Add a ground node with infinite capacitance.
        La température de ce node est obtenue avec TODO
        Si tu lui donne un nom, il va apparaitre dans le graphe.
        """
        pass
        
    # Set resistance between node i and j
    def set_resistance(self, i, j, R):
        """
        Set thermal resistance R [K/W] between nodes i and j.
        """
        if i == j:
            raise ValueError("No self-resistance allowed.")
        self.resistances[(i, j)] = R
        self.resistances[(j, i)] = R  # symmetric

    def set_heated_node(self, node_index, q, condition=lambda t, T: True):
        """
        Créer une condition de chaufage qui injecte q[W] dans le node_index si la condition est True.
        """
        self.heated_nodes.append((node_index, q, condition))

    def build_matrices(self):
        """
        Build capacitance matrix C and conductance matrix G.
        """
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
    def simulate(self, T0=None, dt=120, total_time=108094*60):
        """
        Simule la température des nodes dans le circuit thermique en bonds de 2 minutes.
        T0: Initial temperature vector [°C]. If None, uses experimental data average at t=0.
        """
        # Températures initiales set at experimental data Air average at t=0
        if T0 is None:
            initial_avg = tc.measurement_data_df["Total_Average"][0]
            T0 = np.full(tc.n_nodes, initial_avg)
            T0[n2_outdoor] = self.outdoor_temperature_data[0]  # Outdoor node initial temperature

        C, G = self.build_matrices()
        N_steps = int(total_time / dt)

        T = np.zeros((N_steps + 1, self.n_nodes))
        T[0, :] = T0

        # Precompute C^{-1}
        C_inv = np.linalg.inv(C)

        # Time vector
        time = np.linspace(0, total_time, N_steps + 1)

        # Precompute closest outdoor temperatures to the simulation time steps if available:
        outdoor_temp_full = np.zeros(N_steps + 1)
        for n, t in enumerate(time):
            # find index of the closest outdoor_time
            idx = np.abs(np.array(self.simulation_time_in_seconds) - t).argmin()
            outdoor_temp_full[n] = self.outdoor_temperature_data[idx]

        # Simulation loop
        print('Launching simulation...')
        for n in range(N_steps):
            # Calculmatricielle de la dérivé
            dT = C_inv @ (-G @ T[n, :])

            # Inject heat if condition is met
            for heated_node_index, heating_power, heating_condition in self.heated_nodes:
                if heating_condition(time[n], T[n, :]):
                    dT[heated_node_index] += heating_power / self.capacitances[heated_node_index]

            # T(t+dt) = l'équation
            T[n+1, :] = T[n, :] + dt * dT

            # Update outdoor node
            for outdoor_index in self.outdoor_node_index:
                T[n+1, outdoor_index] = outdoor_temp_full[n+1]

        # Return both T and time vector
        return T, time
    
    def show_temperature_graph(self, T, time, dates_axis=True, compare_with_experimental=True):
        plt.figure(figsize=(8, 5))
        for i in range(self.n_nodes):
            if self.node_names[i]:
                plt.plot(time, T[:, i], label=f"SIMU {self.node_names[i]}")
            else:
                plt.plot(time, T[:, i])
        if dates_axis:
            # Get 15 evenly spaced indices including first and last
            date_ticks = self.simulation_time_in_dates
            indices = np.linspace(0, len(date_ticks) - 1, 15, dtype=int)
            plt.xticks(ticks=time[indices], labels=date_ticks[indices], rotation=45)
            plt.xlabel("Time (dates)")
        else:
            plt.xlabel("Time (seconds)")
        if compare_with_experimental:
            plt.plot(time, self.measurement_data_df["Total_Average"], color='red', linestyle='-', label="DATA OUTDOOR")
        plt.ylabel("Temperature (°C)")
        plt.title("Temperature Evolution of Thermal Nodes")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()
    
# Exemple de Circuit
if __name__ == "__main__":
    tc = ThermalCircuit()

    # Add les nodes avec les capacitances en [J/K]
    n0_air = tc.add_node(C=34400, name="AIR") # AIR
    n1_beton = tc.add_node(C=4.2*10**5, name="BETON") # BETON
    n2_outdoor = tc.add_outdoor_node(name="OUTDOOR") # OUTDOOR
    # TODO add retarded ground node

    # Set les resistances entre les nodes en [K/W]
    tc.set_resistance(n0_air, n1_beton, R=0.125)
    tc.set_resistance(n0_air, n2_outdoor, R=0.055)

    # Les heated node gets q[W] injected si la condition en temps(t) et en température(T) is True
    tc.set_heated_node(n0_air, q=100, condition=lambda t, T: (T[n0_air] < 3) and (t > 0) and (t < 3000000))
    tc.set_heated_node(n0_air, q=100, condition=lambda t, T: (T[n0_air] < 3) and (t > 3500000))

    # Simu
    simulation_temperature_matrix, simulation_time = tc.simulate()

    # Graphe de la simu
    tc.show_temperature_graph(simulation_temperature_matrix, simulation_time)