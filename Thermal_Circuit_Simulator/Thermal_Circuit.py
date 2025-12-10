import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from pathlib import Path
import datetime as dt
from matplotlib.dates import DateFormatter, AutoDateLocator

def get_experimental_data():
    """
    Fetch le data from le CSV du CSV "Dataset of weighing station temperature measurements.csv"
    dans le folder "doi-10.5683-sp3-iaas16" qui se trouve dans le même dossier que ce script python.
    La référence à la path de ce script se fait automatiquement avec os.
    Cette fonction retourne:
    - outdoor_temperature_data: liste des températures extérieures mesurées [°C]
    - simulation_time_in_dates: liste des timestamps des mesures (format datetime)  
    - simulation_time: liste des timestamps des mesures en secondes absolues (float)
    - measurement_data_df: DataFrame pandas contenant toutes les données mesurées, avec des
        colonne supplémentaire:
        - "Low_avg": moyenne des températures basses mesurées
        - "Mid_avg": moyenne des températures moyennes mesurées 
        - "Top_avg": moyenne des températures hautes mesurées
        - "Total_Average": moyenne pondérée des températures mesurées (35% Low, 23% Mid, 41.1% Top)
    """
    script_directory = os.path.dirname(os.path.abspath(__file__))
    irl_data_path = os.path.join(script_directory, "doi-10.5683-sp3-iaas16", "Dataset of weighing station temperature measurements.csv")
    measurement_data_df = pd.read_csv(irl_data_path, sep=';')
    outdoor_temperature_data = measurement_data_df['Outdoor temperature [deg. C]'].to_list()
    simulation_time_in_dates = pd.to_datetime(measurement_data_df['Time'].to_list())
    simulation_time = (simulation_time_in_dates - simulation_time_in_dates[0]).total_seconds()
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
    """
    Classe d'un circuit thermique simulable fait de noeuds avec resistance et capacitance.
    Fonctions pour l'utilisateur:
        add_node(): Ajoute un noeud avec une capacitance thermique C [J/K].
        add_outdoor_node(): Ajoute un noeud exterieur avec C=infini et T=T_ext(always).
        set_heated_node(): Indique un condition et une intensité de chauffage q[W] à un noeud déjà existant.
        set_resistance(): Connecte 2 noeuds avec une résistances thermiques R [K/W].
        simulate(): Simule la température des noeuds dans le temps.
        show_temperature_graph(): Affiche le graphe des températures simulées.
    """
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

    def add_node(self, C, name=None):
        """
        Add a thermal node with capacitance C [J/K].
        Les nodes avec un nom vont apparaitre dans le graphe.
        """
        self.capacitances.append(C)
        self.n_nodes += 1
        self.node_names.append(name)
        return self.n_nodes - 1 # Retourne l'index du node ajouted
    
    def add_outdoor_node(self, name=None):
        """
        Add an outdoor node with infinite capacitance.
        Cette nodes always has T = T_ext.
        Les nodes avec un nom vont apparaitre dans le graphe.
        """
        self.capacitances.append(np.inf)
        self.n_nodes += 1
        self.outdoor_node_index.append(self.n_nodes - 1)
        self.node_names.append(name)
        return self.n_nodes - 1  # Retourne l'index du node ajouted
        
    def set_resistance(self, i, j, R, R_Fan_On=None):
        """
        Set thermal resistance R [K/W] between les nodes i and j.
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
        self.heated_nodes.append((node_index, q, condition))

    def build_matrices(self):
        """
        Build capacitance matrix C and conductance matrix G(avec et sans la convection des aerothermes).
        """
        N = self.n_nodes 
        C = np.diag(self.capacitances)
        G = np.zeros((N, N))
        G_fan_on = np.zeros((N, N))

        # Hors Diagonale
        for Resistance_Connexions in self.resistances.items():
            (i, j), R = Resistance_Connexions
            G[i, j] = -1.0 / R
        for Resistance_fan_on_Connexions in self.resistances_fan_on.items():
            (i, j), R_fan_on = Resistance_fan_on_Connexions
            G_fan_on[i, j] = -1.0 / R_fan_on

        # Diagonale
        for i in range(N): # Diagonale
            G[i, i] = -np.sum(G[i, :])
        for i in range(N):
            G_fan_on[i, i] = -np.sum(G_fan_on[i, :])

        return C, G, G_fan_on
    
    def simulate(self, T0=None, dt=120, total_time=108094*60):
        """
        Simule la température des nodes dans le circuit thermique en bonds de dt.
        T0: Initial temperature vector [°C]. If None, uses experimental data average at t=0.
        """
        # Températures initiales
        if T0 is None:
            initial_avg = self.measurement_data_df["Total_Average"][0]
            T0 = np.full(self.n_nodes, initial_avg)
            for n in self.outdoor_node_index:
                T0[n] = self.outdoor_temperature_data[0]
        
        # Construction des matrices
        C, G, G_fan_on = self.build_matrices()

        # Preparation du vecteur de stockage des températures
        N_steps = int(total_time / dt)
        T = np.zeros((N_steps + 1, self.n_nodes))
        T[0, :] = T0

        # Precompute C^{-1}
        C_inv = np.linalg.inv(C)

        # Time vector
        time = np.linspace(0, total_time, N_steps + 1)

        # Interpolation de la température extérieure du data pour matcher le temps de simulation
        outdoor_temp_full = np.interp(time, self.simulation_time_in_seconds, self.outdoor_temperature_data)

        # heater_on vector pour stocker l'état du chauffage
        heater_on = np.zeros(N_steps + 1, dtype=bool)

        # Simulation loop
        print('Launching simulation...')
        for n in range(N_steps):

            # Calcul de la difference de température pour chaque noeud
            if self.fan_on:
                dT = C_inv @ (G_fan_on @ T[n, :])
            else:
                dT = C_inv @ (G @ T[n, :])

            # Ajout de l'injection de chaleur dans la différence de température pour les noeuds chauffés
            fan_state = False
            for heated_node_index, heating_power, heating_condition in self.heated_nodes:
                if heating_condition(time[n], T[n, :]):
                    dT[heated_node_index] -= heating_power / self.capacitances[heated_node_index]
                    fan_state = True
            self.fan_on = fan_state # Stockage de l'état du fan pour la convection de la prochaine itération
            heater_on[n] = fan_state # Stockage de l'état du chauffage à l'instant n

            # Calcul du prochain pas de temps avec Temp(n+1) = Temp(n) + Temp_diff * dt
            T[n+1, :] = T[n, :] - dt * dT

            # Impose la température extérieure des noeuds extérieurs
            for outdoor_index in self.outdoor_node_index:
                T[n+1, outdoor_index] = outdoor_temp_full[n+1]

        # Copy Heater State pour completer le vecteur heater_on
        heater_on[-1] = heater_on[-2]

        # Calcul du temps total de chauffage total
        dt = time[1] - time[0]
        total_on_time = np.sum(heater_on) * dt
        print(f"Heater ON time = {total_on_time/3600:.2f} hours")

        return T, time, heater_on
    
    def show_temperature_graph(self, T, time, heater_on, dates_axis=True, compare_with_experimental=True, Node_to_compare = 5):
        """
        Prend le output de la fonction simulate().
        Affiche le graphe des températures simulées et print le temps de fonctionnement des aérothermes.
        compare_with_experimental: Si True, ajoute les données expérimentales au graphe et print la Mean Absolute Error.
        Node_to_compare: Index du noeud à comparer avec les données expérimentales(sert à calculer la MAE).
        """
        plt.figure(figsize=(8, 5)) # Main figure

        for i in range(self.n_nodes): # Plot les nodes avec un nom
            if self.node_names[i]:
                plt.plot(time, T[:, i], label=f"{self.node_names[i]} (Simulation)")
                if i in [t[0] for t in self.heated_nodes]:
                    plt.fill_between(time, T[:, i], y2=0, where=heater_on, color='red', alpha=0.1, zorder=2, label="Aérotherme en marche (Simulation)")

        # Formatage des axes
        if dates_axis:
            date_ticks = np.array(self.simulation_time_in_dates)
            n_ticks = 10
            date_idx = np.linspace(0, len(date_ticks) - 1, n_ticks, dtype=int)
            date_labels = [pd.Timestamp(date_ticks[i]).strftime('%Y-%m-%d') for i in date_idx]
            time_pos = np.linspace(time[0], time[-1], n_ticks)
            plt.xticks(ticks=time_pos, labels=date_labels, fontsize=15, rotation=30)
            plt.xlabel("Temps (dates)", fontsize=20)
        else:
            plt.xlabel("Temps (secondes)", fontsize=20)
        plt.ylabel("Température (°C)", fontsize=25)
        plt.yticks(fontsize=20)

        # Calcul de la MAE
        # Devrait être codeé comme une fonction seule mais too bad
        if compare_with_experimental: 
            measurement_data_avg_interpolation = np.interp(time, self.simulation_time_in_seconds, self.measurement_data_df["Total_Average"])
            plt.plot(time, measurement_data_avg_interpolation, color='red', linestyle='-', label="Air intérieur", zorder=1)
            ax = plt.gca()
            ax.vlines(3110400, 0, 1, color='red', linestyle='--', transform=ax.get_xaxis_transform(), label='Fonctionnement normal des aérotherme')
            mae_nan = np.abs(measurement_data_avg_interpolation - T[:, Node_to_compare])
            mae_nan = mae_nan[~np.isnan(mae_nan)]
            mae = np.mean(mae_nan) 
            print(f"Average |Sim - Experimental| = {mae:.2f} °C")

        # Interpolation des temps de la température extérieure pour matcher ceux de la simu
        outdoor_temp_interp = np.interp(time, self.simulation_time_in_seconds, self.outdoor_temperature_data)
        plt.plot(time, outdoor_temp_interp, color='cyan', linestyle='-', label="Extérieur", zorder=2)

        # Final Touches et Display
        # plt.title("Simulation du Circuit Thermique", fontsize=25)
        plt.grid(True)
        plt.legend(loc=3, fontsize=15)
        plt.tight_layout()
        # plt.subplots_adjust( # Moi c'est bon avec ca mais peutêtre que toi tu préfères mettre plt.tight_layout()
        #     left=0.075,
        #     right=0.98,
        #     bottom=0.15,
        #     top=0.98
        # )
        plt.show()
    
# Simulation du puits thermique du poste de pesée
if __name__ == "__main__":
    tc = ThermalCircuit()

    # Add les nodes avec les capacitances en [J/K]
    node_exterieur = tc.add_outdoor_node()
    node_sol_bottom = tc.add_outdoor_node()
    node_sol_side = tc.add_outdoor_node()
    node_isolant_side = tc.add_node(C=2176717.4)
    node_beton_side = tc.add_node(C=1.109*10**8)
    node_air = tc.add_node(C=200000, name="Air Intérieur")
    node_beton_sol = tc.add_node(C=1.1379*10**8)
    node_plaque = tc.add_node(C=6.3*10**6)

    # Set les resistances entre les nodes en [K/W]
    tc.set_resistance(node_sol_side, node_isolant_side, R=0.0775)
    tc.set_resistance(node_isolant_side, node_beton_side, R=0.0783525)
    tc.set_resistance(node_air, node_beton_side, R=0.0032785, R_Fan_On=0.001641)
    tc.set_resistance(node_air, node_beton_sol, R=0.01611835, R_Fan_On=0.00179635)
    tc.set_resistance(node_beton_sol, node_sol_bottom, R=0.00071835)
    tc.set_resistance(node_air, node_plaque, R=0.0021578355, R_Fan_On=0.0010790355)
    tc.set_resistance(node_air, node_exterieur, R=2.595)
    tc.set_resistance(node_plaque, node_exterieur, R=0.0013370355)

    # Les heated node gets q[W] injected si la condition en temps(t) et en température(T) is True
    efficacite_thermique=0.44 # en [0 à 1]
    puissance_full = 60000 # en [kW]
    puissance_not_full = 50000 # en [kW]
    tc.set_heated_node(node_air, q=puissance_not_full*efficacite_thermique, condition=lambda t, T: ((T[node_exterieur] < 0) and (t < 3110400)))
    tc.set_heated_node(node_air, q=puissance_full*efficacite_thermique, condition=lambda t, T: ((T[node_exterieur] < 0) and (t > 3110400)))
 
    # Simulation
    Temp, Time, Heater = tc.simulate()

    # Graph de la simulation
    tc.show_temperature_graph(Temp, Time, Heater)