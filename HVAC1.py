import pulp
import numpy as np
import matplotlib.pyplot as plt

# Escenario Base (setpoint fijo)
def solve_hvac_setpoint_fixed(N, T_out, price_buy, T0, setpoint, c, b_heat, b_cool):
    model = pulp.LpProblem("HVAC_Base_Setpoint", pulp.LpMinimize)
    x_heat = pulp.LpVariable.dicts("x_heat", range(N), lowBound=0)
    x_cool = pulp.LpVariable.dicts("x_cool", range(N), lowBound=0)
    T_in   = pulp.LpVariable.dicts("T_in", range(N))
    model.addConstraint(T_in[0] == T0)
    for t in range(N-1):
        model.addConstraint(
            T_in[t+1] == (1 - c)*T_in[t] + c*T_out[t] + b_heat*x_heat[t] - b_cool*x_cool[t]
        )
    for t in range(N):
        model.addConstraint(T_in[t] == setpoint)
    model.setObjective(
        pulp.lpSum(price_buy[t] * (x_heat[t] + x_cool[t]) for t in range(N))
    )
    model.solve(pulp.PULP_CBC_CMD(msg=0))
    xh = np.array([x_heat[t].varValue for t in range(N)])
    xc = np.array([x_cool[t].varValue for t in range(N)])
    Tin = np.array([T_in[t].varValue for t in range(N)])
    cost = np.sum(price_buy * (xh + xc))
    return xh, xc, Tin, cost

# Escenario con banda
def solve_hvac_heat_cool(N, T_out, price_buy, T0, T_min, T_max, c, b_heat, b_cool):
    model = pulp.LpProblem("HVAC_Band_Comfort", pulp.LpMinimize)
    x_heat = pulp.LpVariable.dicts("x_heat", range(N), lowBound=0)
    x_cool = pulp.LpVariable.dicts("x_cool", range(N), lowBound=0)
    T_in   = pulp.LpVariable.dicts("T_in", range(N))
    model.addConstraint(T_in[0] == T0)
    for t in range(N-1):
        model.addConstraint(
            T_in[t+1] == (1 - c)*T_in[t] + c*T_out[t] + b_heat*x_heat[t] - b_cool*x_cool[t]
        )
    for t in range(N):
        model.addConstraint(T_in[t] >= T_min)
        model.addConstraint(T_in[t] <= T_max)
    model.setObjective(
        pulp.lpSum(price_buy[t] * (x_heat[t] + x_cool[t]) for t in range(N))
    )
    model.solve(pulp.PULP_CBC_CMD(msg=0))
    xh = np.array([x_heat[t].varValue for t in range(N)])
    xc = np.array([x_cool[t].varValue for t in range(N)])
    Tin = np.array([T_in[t].varValue for t in range(N)])
    cost = np.sum(price_buy * (xh + xc))
    return xh, xc, Tin, cost

# Escenario min/max
def solve_hvac_minmax_only(N, T_out, price_buy, T0, T_min, T_max, c, b_heat, b_cool):
    model = pulp.LpProblem("HVAC_Boundary_Only", pulp.LpMinimize)
    x_heat = pulp.LpVariable.dicts("x_heat", range(N), lowBound=0)
    x_cool = pulp.LpVariable.dicts("x_cool", range(N), lowBound=0)
    z = pulp.LpVariable.dicts("z", range(N), cat=pulp.LpBinary)
    T_in = pulp.LpVariable.dicts("T_in", range(N))
    model.addConstraint(
        T_in[0] == T_min + (T_max - T_min)*z[0]
    )
    for t in range(N-1):
        model.addConstraint(
            T_in[t] == T_min + (T_max - T_min)*z[t]
        )
        model.addConstraint(
            T_in[t+1] == (1 - c)*T_in[t] + c*T_out[t] + b_heat*x_heat[t] - b_cool*x_cool[t]
        )
    model.addConstraint(
        T_in[N-1] == T_min + (T_max - T_min)*z[N-1]
    )
    model.setObjective(
        pulp.lpSum(price_buy[t] * (x_heat[t] + x_cool[t]) for t in range(N))
    )
    model.solve(pulp.PULP_CBC_CMD(msg=0))
    xh = np.array([x_heat[t].varValue for t in range(N)])
    xc = np.array([x_cool[t].varValue for t in range(N)])
    Tin = np.array([T_in[t].varValue for t in range(N)])
    cost = np.sum(price_buy * (xh + xc))
    zz = np.array([z[t].varValue for t in range(N)])
    return xh, xc, Tin, cost, zz

# Datos
N = 24
T_out = np.array([
    15, 14, 14, 13, 12, 12, 13, 15, 18, 21, 24, 26,
    28, 28, 28, 27, 25, 23, 22, 21, 20, 19, 18, 17
])
price_buy = np.array([
    0.08, 0.07, 0.06, 0.06, 0.05, 0.05, 0.06, 0.08,
    0.11, 0.14, 0.18, 0.19, 0.20, 0.20, 0.18, 0.16,
    0.15, 0.15, 0.14, 0.12, 0.10, 0.09, 0.08, 0.07
])
T0 = 22.0
c = 0.3
b_heat = 0.5
b_cool = 0.4

# Escenario Base
xhBASE, xcBASE, TinBASE, costBASE = solve_hvac_setpoint_fixed(
    N, T_out, price_buy, T0, 22.0, c, b_heat, b_cool
)

# Escenario A
xhA, xcA, TinA, costA = solve_hvac_heat_cool(
    N, T_out, price_buy, T0, 21.5, 24.0, c, b_heat, b_cool
)

# Escenario B
xhB, xcB, TinB, costB = solve_hvac_heat_cool(
    N, T_out, price_buy, T0, 20.0, 25.0, c, b_heat, b_cool
)

# Escenario C
xhC, xcC, TinC, costC, zzC = solve_hvac_minmax_only(
    N, T_out, price_buy, T0, 20.0, 25.0, c, b_heat, b_cool
)

# Resultados
print("Base:", costBASE)
print("A:", costA)
print("B:", costB)
print("C:", costC)

# Gráficas
horas = np.arange(N)

plt.figure()
plt.plot(horas, TinBASE, label="Base (°C)")
plt.plot(horas, TinA, label="A (°C)")
plt.plot(horas, TinB, label="B (°C)")
plt.plot(horas, TinC, label="C (°C)")
plt.plot(horas, T_out, "--", label="T_out (°C)")
plt.xlabel("Hora")
plt.ylabel("Temperatura (°C)")
plt.title("Perfil de Temperaturas")
plt.legend()

plt.figure()
plt.plot(horas, xhBASE, label="Heat Base (kW)")
plt.plot(horas, xhA, label="Heat A (kW)")
plt.plot(horas, xhB, label="Heat B (kW)")
plt.plot(horas, xhC, label="Heat C (kW)")
plt.xlabel("Hora")
plt.ylabel("Potencia (kW)")
plt.title("Calefacción")
plt.legend()

plt.figure()
plt.plot(horas, xcBASE, label="Cool Base (kW)")
plt.plot(horas, xcA, label="Cool A (kW)")
plt.plot(horas, xcB, label="Cool B (kW)")
plt.plot(horas, xcC, label="Cool C (kW)")
plt.xlabel("Hora")
plt.ylabel("Potencia (kW)")
plt.title("Refrigeración")
plt.legend()

pBASE = xhBASE + xcBASE
pA = xhA + xcA
pB = xhB + xcB
pC = xhC + xcC

diffA = pA - pBASE
diffB = pB - pBASE
diffC = pC - pBASE

plt.figure()
plt.bar(horas, diffA)
plt.xlabel("Hora")
plt.ylabel("ΔPotencia (kW)")
plt.title("A vs Base")
plt.figure()
plt.bar(horas, diffB)
plt.xlabel("Hora")
plt.ylabel("ΔPotencia (kW)")
plt.title("B vs Base")
plt.figure()
plt.bar(horas, diffC)
plt.xlabel("Hora")
plt.ylabel("ΔPotencia (kW)")
plt.title("C vs Base")

plt.show()
