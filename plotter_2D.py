import pandas as pd
import math
import matplotlib.pyplot as plt
import os
import re
import numpy as np
from lib import search_for_export_csv, extract_parameters_by_file_name

# ---------------------------------- Config ---------------------------------- #
save_file_name = r"2D_omega"
save_file_extension = ".svg"

do_plot_line_from_origin = False

is_multi_files = False
target_folder_multi_files = "/csv/1deg/"
plots_folder = "plots"

# doesn't need to update the parameters if individual file is used
parameter_dict = {
    "eps": 0.0177,
    "epsphi": 0.0,
    "kappa": 1.0,
    "deltas": 1.0,
    "beta": 90.0,
    "alpha": 30.0,
    "theta": 30.0,
    "time": 100,
}

simulation_time = 100
method = "Feagin14 Method"
# ------------------------------------ --- ----------------------------------- #


def tangent_calculator(x_, y_):
    return y_ / x_


def point_finder(x_i, y_i, x_f):
    b = 0
    m = tangent_calculator(x_i, y_i)
    y_f = x_f * m + b

    xs = [x_i, x_f]
    ys = [y_i, y_f]
    return xs, ys


def Plotter(save_filename):
    global parameter_dict

    # get the list of all csv files
    if is_multi_files:
        path_ = os.path.dirname(__file__) + target_folder_multi_files
        filelst = os.listdir(path_)
    else:
        path_ = ""
        filelst = indivisual_file_names_to_read

    parameter_dict = extract_parameters_by_file_name(filelst[0])

    eps = parameter_dict["eps"]
    epsphi = parameter_dict["epsphi"]
    kappa = parameter_dict["kappa"]
    deltas = parameter_dict["deltas"]
    beta = parameter_dict["beta"]
    alpha = parameter_dict["alpha"]
    theta = parameter_dict["theta"]
    simulation_time = parameter_dict["time"]

    for fname in filelst:
        data = pd.read_csv(path_ + fname)
        df = pd.DataFrame(data, columns=["timestamp", "omega_rho", "omega_z", "rho", "z", "drho", "dz"])
        y_ = adiabtic_calculator(df)

        x_ = df["z"]
        #y_ = df["omega_rho"]/df["omega_z"]
        y_ = df["rho"]

        if is_multi_files:
            parameter_dict = extract_parameters_by_file_name(fname)
            linename = "approx " if "approx" in fname else "exact" 
            plt.plot(x_, y_, label=linename + "→ " + r"$\theta$=" + str(parameter_dict["theta"]))
        else:
            plt.plot(x_, y_)

    if do_plot_line_from_origin:
        xp_, yp_ = point_finder(x_.iloc[0], y_.iloc[0], x_.iloc[-1])
        plt.plot(xp_, yp_, "--", label=f"line passes origin")

    plt.rcParams["figure.dpi"] = 150
    plt.ylabel(r"$\widetilde{R}$")
    plt.xlabel(r"$\tau$")
    plt.suptitle(r"$\widetilde{R}$ vs $\tau$", fontsize=12)
    plt.title(
        #f"$\\theta_0 = {theta}^{{\circ}}$ , $\\alpha_0={alpha}^{{\circ}}$ , $\\beta_0 = {beta}^{{\circ}}$, $\\phi_0 = 0.0^{{\circ}}$, $\\kappa = {kappa}$, $\\delta_* = {deltas}$, $\\epsilon_ = {eps}$",
        f"$\\theta_0 = {theta}^{{\circ}}$ , $\\alpha_0={alpha}^{{\circ}}$ , $\\beta_0 = {beta}^{{\circ}}$, $\\phi_0 = 0.0^{{\circ}}$, $\\delta_* = {deltas}$, $\\epsilon_\phi = {epsphi}$, $\\epsilon = {eps}$",
        loc="right",
        fontsize=8,
        color="grey",
        style="italic",
    )
    plt.title(
        f"{method}, $\\tau = {simulation_time}$",
        loc="left",
        fontsize=8,
        color="grey",
        style="italic",
    )
    
    if is_multi_files:
        plt.legend()
    plt.tight_layout()
    path_to_save = os.path.join(plots_folder, str(save_filename + save_file_extension))
    plt.savefig(path_to_save, dpi=600)
    path_to_save = os.path.join(plots_folder, str(save_filename + ".png"))
    plt.savefig(path_to_save, dpi=600)
    plt.show()


def adiabtic_calculator(df):
    velocity = df[["drho", "dz"]]
    position = df[["rho", "z"]]

    velocity_2 = velocity **2
    print(velocity['drho'].shape)

    integrate = velocity['drho'] * position['z'] - velocity['dz'] * position['rho']
    integrate_2 = velocity_2['drho'] * position['z'] - velocity_2['dz'] * position['rho']

    # Compute the changes in the components of X
    delta_X = np.diff(position, axis=0)

    print(velocity.shape)

    adiabatic = np.cumsum(velocity.iloc[:, 0].values * np.append(delta_X[:, 0], 0))



    # Compute the integral of V.dX as a vector at each time step
    integral_VdX_vector = np.cumsum(np.sum(velocity[:-1] * delta_X, axis=1))

    print(adiabatic.shape)

    plt.plot(df['timestamp'], adiabatic)
    #plt.plot(df['timestamp'], integrate)
    #plt.plot( df['timestamp'], integrate_2)

    plt.show()

    return adiabatic




if not is_multi_files:
    chosen_csv = search_for_export_csv()
    indivisual_file_names_to_read = [chosen_csv]
else:
    chosen_csv = "multi_plot"
Plotter(chosen_csv.replace(".csv", ""))
# extract_parameters_by_file_name(indivisual_file_names_to_read[0])
