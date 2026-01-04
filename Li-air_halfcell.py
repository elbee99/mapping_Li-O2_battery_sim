import pybamm
import numpy as np
import matplotlib.pyplot as plt

model = pybamm.BaseModel()

# potentials
phi = pybamm.Variable("Positive electrode potential [V]", domain="positive electrode")
phi_e_s = pybamm.Variable("Separator electrolyte potential [V]", domain="separator")
phi_e_p = pybamm.Variable(
    "Positive electrolyte potential [V]", domain="positive electrode"
)
phi_e = pybamm.concatenation(phi_e_s, phi_e_p)

# dissolved species concentrations
c_Li_p = pybamm.Variable("Positive electrode Li+ concentration [mol.m-3]", domain="positive electrode")
c_Li_s = pybamm.Variable("Separator Li+ concentration [mol.m-3]", domain="separator")
c_Li = pybamm.concatenation(c_Li_s, c_Li_p)

c_O2_p = pybamm.Variable("Positive electrode O2 concentration [mol.m-3]", domain="positive electrode")
c_O2_s = pybamm.Variable("Separator O2 concentration [mol.m-3]", domain="separator")
c_O2 = pybamm.concatenation(c_O2_s, c_O2_p)

c_Li2O2_p = pybamm.Variable("Positive electrode Li2O2 concentration [mol.m-3]", domain="positive electrode")
c_Li2O2_s = pybamm.Variable("Separator Li2O2 concentration [mol.m-3]", domain="separator")
c_Li2O2 = pybamm.concatenation(c_Li2O2_s, c_Li2O2_p)

# fundamental constants
F = pybamm.Parameter("Faraday constant [C.mol-1]")
R = pybamm.Parameter("Molar gas constant [J.mol-1.K-1]")
T = pybamm.Parameter("Temperature [K]")

# material parameters
a0 = pybamm.Parameter("Surface area per unit volume [m-1]")
L_s = pybamm.Parameter("Separator thickness [m]")
L_p = pybamm.Parameter("Positive electrode thickness [m]")
A = pybamm.Parameter("Electrode cross-sectional area [m2]")
sigma = pybamm.Parameter("Positive electrode conductivity [S.m-1]")
kappa = pybamm.Parameter("Electrolyte conductivity [S.m-1]")
eps_0 = pybamm.Parameter("Initial porosity")
eps_min = pybamm.Parameter("Minimum porosity")

# species parameters
D_Li = pybamm.Parameter("Li+ Diffusion coefficient [m2.s-1]")
D_O2 = pybamm.Parameter("O2 Diffusion coefficient [m2.s-1]")
M_Li2O2 = pybamm.Parameter("Li2O2 molar mass [kg.mol-1]")
rho_Li2O2 = pybamm.Parameter("Li2O2 density [kg.m-3]")
tplus = pybamm.Parameter("Li+ transference number")
kappa_e = pybamm.Parameter("Electrolyte conductivity [S.m-1]")
dlnfdlnc = pybamm.Parameter("Activity dependance")

# system initial conditions
I_app = pybamm.Parameter("Applied current [A]")
c_Li_0 = pybamm.Parameter("Initial Li+ concentration [mol.m-3]")
c_O2_0 = pybamm.Parameter("Initial O2 concentration [mol.m-3]")

# Calculate Li2O2 volume fraction
eps_Li2O2_p = c_Li2O2_p * M_Li2O2 / rho_Li2O2
eps_Li2O2_s = c_Li2O2_s * M_Li2O2 / rho_Li2O2
eps_Li2O2 = pybamm.concatenation(eps_Li2O2_s, eps_Li2O2_p)

# Current porosity (decreases as Li2O2 forms)
eps_p = eps_0 - eps_Li2O2_p
eps_s = eps_0 - eps_Li2O2_s
eps = pybamm.concatenation(eps_s, eps_p)

# Specific surface area (decreases with Li2O2 formation)
a_p = a0 * (1 - pybamm.maximum(((eps_Li2O2_p)/(eps_0-eps_min)), 1e-10)**0.5)
a_s = a0 * (1 - pybamm.maximum(((eps_Li2O2_s)/(eps_0-eps_min)), 1e-10)**0.5)
a = pybamm.concatenation(a_s, a_p)

# electrochemical parameters
inputs = {"Positive electrode O2 concentration": c_O2_p, "Positive electrode Li concentration": c_Li_p}
j0 = pybamm.FunctionParameter(
    "Positive electrode exchange-current density [A.m-2]", inputs
)
U = pybamm.Parameter("Positive electrode OCP [V]")

# interfacial current density
j_s = pybamm.PrimaryBroadcast(0, "separator")
j_p = 2 * j0 * pybamm.sinh((F / R / T) * (phi - phi_e_p - U))
j = pybamm.concatenation(j_s, j_p)

# charge conservation equations
i = -sigma * pybamm.grad(phi)
i_e = -kappa * pybamm.grad(phi_e) - (2*R*T*kappa_e/F) * (1-tplus) * (1+dlnfdlnc) * pybamm.grad(c_Li)
model.algebraic = {
    phi: pybamm.div(i) + a_p * j_p,
    phi_e: pybamm.div(i_e) - a * j,
}


# Effective diffusivities (Bruggeman correction)
D_Li_eff = D_Li * eps**1.5
D_O2_eff = D_O2 * eps**1.5

# Fluxes
N_Li = -D_Li_eff * pybamm.grad(c_Li) + (i_e * tplus) / F
N_O2 = -D_O2_eff * pybamm.grad(c_O2)

# Mass conservation for Li+ 
depsdt_Li = -a * j / F * M_Li2O2 / rho_Li2O2  # from Li2O2 formation
d_eps_c_Li_dt = -pybamm.div(N_Li) + 2 * a * j /(2 * F)
dc_Li_dt = (d_eps_c_Li_dt - c_Li * depsdt_Li) / eps
model.rhs[c_Li] = dc_Li_dt

# Mass conservation for O2
depsdt_O2 = -a * j / F * M_Li2O2 / rho_Li2O2  # same porosity change
d_eps_c_O2_dt = -pybamm.div(N_O2) + a * j /(2 * F)
dc_O2_dt = (d_eps_c_O2_dt - c_O2 * depsdt_O2) / eps
model.rhs[c_O2] = dc_O2_dt

# Li2O2 formation rate (direct from reaction)
dc_Li2O2_dt = -a * j /(2 * F)
model.rhs[c_Li2O2] = dc_Li2O2_dt

# boundary conditions
model.boundary_conditions = {
    phi: {
        "left": (pybamm.Scalar(0), "Neumann"),
        "right": (-I_app / A / sigma, "Neumann"),
    },
    phi_e: {
        "left": (pybamm.Scalar(0), "Dirichlet"),
        "right": (pybamm.Scalar(0), "Neumann"),
    },
    c_Li: {"left": (c_Li_0, "Dirichlet"), "right": (0, "Neumann")},
    c_O2: {"left": (pybamm.Scalar(0), "Neumann"), "right": (c_O2_0, "Dirichlet")},
    c_Li2O2: {"left": (pybamm.Scalar(0), "Neumann"), "right": (pybamm.Scalar(0), "Neumann")},
}

# initial conditions
model.initial_conditions = {phi: U, phi_e: 0, c_Li: c_Li_0, c_O2: c_O2_0, c_Li2O2: pybamm.Scalar(0)}

model.variables = {
    "Positive electrode potential [V]": phi,
    "Electrolyte potential [V]": phi_e,
    "Positive electrode Li+ concentration [mol.m-3]": c_Li,
    "Positive electrode O2 concentration [mol.m-3]": c_O2,
    "Positive electrode interfacial current density [A.m-2]": j_p,
    "Positive electrode OCP [V]": pybamm.boundary_value(U, "right"),
    "Voltage [V]": pybamm.boundary_value(phi, "right"),
    "Positive electrode Li2O2 concentration [mol.m-3]": c_Li2O2,
    "Positive electrode Li2O2 volume fraction": eps_Li2O2,
    "Porosity": eps,
    "Specific surface area [m-1]": a,
}

def exchange_current_density(c_Li_p, c_O2_p):
    k = 6 * 10 ** (-12) 
    return k * c_Li_p**2 * c_O2_p

param = pybamm.ParameterValues(
    {
        "Surface area per unit volume [m-1]": 2e7,
        "Separator thickness [m]": 50e-6,
        "Positive electrode thickness [m]": 200e-6,
        "Electrode cross-sectional area [m2]": 1e-4,
        "Applied current [A]": 0.5e-3,
        "Positive electrode conductivity [S.m-1]": 10,
        "Electrolyte conductivity [S.m-1]": 1,
        "Li+ Diffusion coefficient [m2.s-1]": 2.11e-9,
        "O2 Diffusion coefficient [m2.s-1]": 0.7e-9,
        "Li+ transference number": 0.25,
        "Activity dependance": -1,
        "Electrolyte conductivity [S.m-1]": 0.1,
        "Faraday constant [C.mol-1]": 96485,
        "Molar gas constant [J.mol-1.K-1]": 8.314,
        "Temperature [K]": 298.15,
        "Maximum concentration in positive electrode [mol.m-3]": 51217,
        "Positive electrode exchange-current density [A.m-2]": exchange_current_density,
        "Positive electrode OCP [V]": 2.96,
        "Initial Li+ concentration [mol.m-3]": 1000,
        "Initial O2 concentration [mol.m-3]": 3.7,
        "Li2O2 molar mass [kg.mol-1]": 45.88e-3,
        "Li2O2 density [kg.m-3]": 2310,
        "Initial porosity": 0.8,
        "Minimum porosity": 0.0
    }
)

# Geometry setup
x_s = pybamm.SpatialVariable("x_s", domain=["separator"], coord_sys="cartesian")
x_p = pybamm.SpatialVariable("x_p", domain=["positive electrode"], coord_sys="cartesian")

geometry = {
    "separator": {x_s: {"min": -L_s, "max": 0}},
    "positive electrode": {x_p: {"min": 0, "max": L_p}},
}

param.process_model(model)
param.process_geometry(geometry)

submesh_types = {
    "separator": pybamm.Uniform1DSubMesh,
    "positive electrode": pybamm.Uniform1DSubMesh,
    "positive particle": pybamm.Uniform1DSubMesh,
}
var_pts = {x_s: 20, x_p: 150}
mesh = pybamm.Mesh(geometry, submesh_types, var_pts)

spatial_methods = {
    "separator": pybamm.FiniteVolume(),
    "positive electrode": pybamm.FiniteVolume(),
}
disc = pybamm.Discretisation(mesh, spatial_methods)
disc.process_model(model)

# solve
solver = pybamm.CasadiSolver(root_tol=1e-2)
hours = 1000
t_eval = np.linspace(0, 3600*hours, 60*hours)
# larger time steps needed for long simulation

# termination condition
min_voltage = pybamm.Scalar(2.0)
termination = pybamm.min(model.variables["Voltage [V]"] - min_voltage)
model.events.append(pybamm.Event("Minimum voltage", termination))

solution = solver.solve(model, t_eval)

print("Solution completed successfully!")
print(f"Final time: {solution.t[-1]:.2f} s")
print(f"Final voltage: {solution['Voltage [V]'].entries[-1]:.3f} V")
# plot dynamic_plot(
pybamm.dynamic_plot(
    solution,
    [
        "Electrolyte potential [V]",
        "Positive electrode interfacial current density [A.m-2]",
        "Positive electrode Li+ concentration [mol.m-3]",
        "Positive electrode O2 concentration [mol.m-3]",
        ["Positive electrode OCP [V]", "Voltage [V]"],
        "Positive electrode Li2O2 volume fraction"
    ],
)

# save solution
save_path = r"example_solutions\DMSO_0p5mAcm2.pkl"
solution.save(save_path)