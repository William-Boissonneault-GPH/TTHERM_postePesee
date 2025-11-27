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
        self.resistances_fan_on = {}
        self.heated_nodes = []
        self.outdoor_node_index = []
        self.node_names = []
        self.n_nodes = 0
        self.fan_on = False

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
    def set_resistance(self, i, j, R, R_Fan_On=None):
        """
        Set thermal resistance R [K/W] between nodes i and j.
        """
        if R_Fan_On == None:
            R_Fan_On = R
        if i == j:
            raise ValueError("No self-resistance allowed.")
        self.resistances[(i, j)] = R
        self.resistances[(j, i)] = R  # symmetric
        self.resistances_fan_on[(i, j)] = R_Fan_On
        self.resistances_fan_on[(j, i)] = R_Fan_On  # symmetric

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
        G_fan_on = np.zeros((N, N))

        for Resistance_Connexions in self.resistances.items():
            (i, j), R = Resistance_Connexions
            G[i, j] = -1.0 / R

        for Resistance_fan_on_Connexions in self.resistances_fan_on.items():
            (i, j), R_fan_on = Resistance_fan_on_Connexions
            G_fan_on[i, j] = -1.0 / R_fan_on

        # Diagonal elements = sum of outgoing conductances
        for i in range(N):
            G[i, i] = -np.sum(G[i, :])  # row must sum to zero
        for i in range(N):
            G_fan_on[i, i] = -np.sum(G_fan_on[i, :])  # row must sum to zero

        return C, G, G_fan_on
    
    # Simulate with explicit Euler
    def simulate(self, T0=None, dt=120, total_time=108094*60):
        """
        Simule la température des nodes dans le circuit thermique en bonds de 2 minutes.
        T0: Initial temperature vector [°C]. If None, uses experimental data average at t=0.
        """
        # Températures initiales set at experimental data Air average at t=0
        if T0 is None:
            initial_avg = self.measurement_data_df["Total_Average"][0]
            T0 = np.full(self.n_nodes, initial_avg)
            for n in self.outdoor_node_index:
                T0[n] = self.outdoor_temperature_data[0]  # Outdoor node initial temperature

        C, G, G_fan_on = self.build_matrices()
        N_steps = int(total_time / dt)

        T = np.zeros((N_steps + 1, self.n_nodes))
        T[0, :] = T0

        # Precompute C^{-1}
        C_inv = np.linalg.inv(C)

        # Time vector
        time = np.linspace(0, total_time, N_steps + 1)

        # Interpolates closest outdoor temperatures to the simulation time steps
        outdoor_temp_full = np.interp(time, self.simulation_time_in_seconds, self.outdoor_temperature_data)

        # heater_on vector to keep track of heater action time
        heater_on = np.zeros(N_steps + 1, dtype=bool)

        # Simulation loop
        print('Launching simulation...')
        for n in range(N_steps):

            # Calculmatricielle de la dérivé
            if self.fan_on:
                dT = C_inv @ (-G_fan_on @ T[n, :])
            else:
                dT = C_inv @ (-G @ T[n, :])

            # Condition d'injection de la chaleur
            fan_state = False
            for heated_node_index, heating_power, heating_condition in self.heated_nodes:
                if heating_condition(time[n], T[n, :]):
                    dT[heated_node_index] += heating_power / self.capacitances[heated_node_index]
                    fan_state = True
            self.fan_on = fan_state
            heater_on[n] = fan_state

            # T(t+dt) = l'équation
            T[n+1, :] = T[n, :] + dt * dT

            # Update outdoor node
            for outdoor_index in self.outdoor_node_index:
                T[n+1, outdoor_index] = outdoor_temp_full[n+1]

        # Last step: copy heater state
        heater_on[-1] = heater_on[-2]

        # Return both T and time vector
        return T, time, heater_on
    
    def show_temperature_graph(self, T, time, heater_on, dates_axis=True, compare_with_experimental=True):
        plt.figure(figsize=(8, 5))
        for i in range(self.n_nodes):
            if self.node_names[i]:
                plt.plot(time, T[:, i], label=f"{self.node_names[i]} (Simulation)")
                if i in [t[0] for t in self.heated_nodes]:
                    plt.fill_between(time, T[:, i], y2=0, where=heater_on, color='red', alpha=0.1, zorder=2, label="Aérotherme en marche (Simulation)")
        if dates_axis:
            date_ticks = self.simulation_time_in_dates
            indices = np.linspace(0, len(date_ticks) - 1, 10, dtype=int)
            date_labels = [date_ticks[i].strftime('%Y-%m-%d') for i in indices]
            plt.xticks(ticks=time[indices], labels=date_labels)
            plt.xlabel("Temps (dates)")



        else:
            plt.xlabel("Temps (secondes)")
        if compare_with_experimental:
            measurement_data_avg_interpolation = np.interp(time, self.simulation_time_in_seconds, self.measurement_data_df["Total_Average"])
            plt.plot(time, measurement_data_avg_interpolation, color='red', linestyle='-', label="Air intérieur", zorder=1)
            ax = plt.gca()
            ax.vlines(3110400, 0, 1, color='red', linestyle='--', transform=ax.get_xaxis_transform(), label='Fonctionnement normal des aérotherme')
        
        # Interpolate outdoor temperature to simulation time
        outdoor_temp_interp = np.interp(time, self.simulation_time_in_seconds, self.outdoor_temperature_data)
        plt.plot(time, outdoor_temp_interp, color='cyan', linestyle='-', label="Extérieur", zorder=2)

        plt.ylabel("Température (°C)")
        plt.grid(True)
        plt.legend(loc=3)
        plt.tight_layout()
        plt.show()
    
# Exemple de Circuit
if __name__ == "__main__":
    tc = ThermalCircuit()

    # Add les nodes avec les capacitances en [J/K]
    node_exterieur = tc.add_outdoor_node()
    node_sol_bottom = tc.add_outdoor_node()
    node_sol_side = tc.add_outdoor_node()
    node_isolant_side = tc.add_node(C=42411)#ADDED C ne 0 
    node_beton_side = tc.add_node(C=1.109*10**8)
    node_air = tc.add_node(C=200000, name="Air Intérieur")
    node_beton_sol = tc.add_node(C=1.1379*10**8)
    node_plaque = tc.add_node(C=6.3*10**6)
    # TODO add retarded ground node

    # Set les resistances entre les nodes en [K/W]
    tc.set_resistance(node_sol_side, node_isolant_side, R=0.0775)
    tc.set_resistance(node_isolant_side, node_beton_side, R=0.0783525)
    tc.set_resistance(node_air, node_beton_side, R=0.0032785, R_Fan_On=0.001641)#ADDED R est la somme de Beton->Thermocouple->Air
    tc.set_resistance(node_air, node_beton_sol, R=0.01611835, R_Fan_On=0.00179635)
    tc.set_resistance(node_beton_sol, node_sol_bottom, R=0.00071835)
    tc.set_resistance(node_air, node_plaque, R=0.0021578355, R_Fan_On=0.0010790355)
    tc.set_resistance(node_air, node_exterieur, R=2.595)
    tc.set_resistance(node_plaque, node_exterieur, R=0.0013370355)

    # # Les heated node gets q[W] injected si la condition en temps(t) et en température(T) is Trues
    tc.set_heated_node(node_air, q=36000*0.60, condition=lambda t, T: ((T[node_exterieur] < 0) and (t < 3110400)))
    tc.set_heated_node(node_air, q=46000*0.60, condition=lambda t, T: ((T[node_exterieur] < 0) and (t > 3110400)))
    # TODO find les bonnes conditions et les bonnes puissances

    # Simu
    Temp, Time, Heater = tc.simulate()

    # Graphe de la simu
    tc.show_temperature_graph(Temp, Time, Heater)


"""
Considération:
-J'ai ajouté une capacitance à l'isolant
-J'ai merger les Resistance avant et après le thermocouple pour pouvoir enlever le thermocouple
-Pour l'instant T_sol = T_Ext
-Pour l'instant Heater On si T_ext<0
-Je considère que le Heater son efficaces à 60% (pour injection de chaleur)
"""