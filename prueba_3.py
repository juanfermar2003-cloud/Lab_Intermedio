import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter

# ===============================
# DATOS (GLICERINA / H) - GUÍA
# ===============================
B0_1 = np.array([358, 369, 384, 396, 409, 420, 433, 442], dtype=float)  # mT
f_1  = np.array([16.01, 16.50, 17.02, 17.51, 18.01, 18.50, 19.01, 19.47], dtype=float)  # MHz

# ===============================
# DATOS (PTFE / F) - GUÍA
# ===============================
B0_2 = np.array([372, 386, 396, 409, 422, 434, 445, 459], dtype=float)  # mT
f_2  = np.array([16.00, 16.50, 17.01, 17.50, 18.01, 19.01, 19.00, 19.48], dtype=float)  # MHz

# ===============================
# INCERTIDUMBRES INSTRUMENTALES
# ===============================
sigma_B0 = 1      # mT
sigma_f  = 0.01   # MHz

# ==========================================================
# AJUSTE LINEAL (MÍNIMOS CUADRADOS SIN PESOS)
# y = m x + b
# ==========================================================
def ajuste_lineal_minimos_cuadrados(x, y):
    n = len(x)

    Sx  = np.sum(x)
    Sy  = np.sum(y)
    Sxx = np.sum(x**2)
    Sxy = np.sum(x*y)

    D = n*Sxx - (Sx**2)

    m = (n*Sxy - Sx*Sy) / D
    b = (Sy*Sxx - Sx*Sxy) / D

    # Predicción y residuales
    y_hat = m*x + b
    r = y - y_hat

    # Varianza residual
    Sr2 = np.sum(r**2)
    s2  = Sr2 / (n - 2)
    s   = np.sqrt(s2)

    # Incertidumbres (sin pesos)
    delta_m = np.sqrt(n*Sr2 / ((n - 2)*D))
    delta_b = np.sqrt((1 + (Sx**2)/D) * (Sr2 / (n*(n - 2))))

    return m, b, delta_m, delta_b, r, y_hat, s

# ===============================
# APLICAR A CADA CONJUNTO
# ===============================
a1, c1, sigma_a1, sigma_c1, r1, f1_hat, s1 = ajuste_lineal_minimos_cuadrados(B0_1, f_1)
a2, c2, sigma_a2, sigma_c2, r2, f2_hat, s2 = ajuste_lineal_minimos_cuadrados(B0_2, f_2)

# ===============================
# IMPRESIÓN
# ===============================
print("\n===============================")
print("AJUSTE LINEAL (a mano, sin polyfit)")
print("===============================")
print(f"Glicerina (H) - Guía: a = {a1:.6f} ± {sigma_a1:.6f}  [MHz/mT],  c = {c1:.6f} ± {sigma_c1:.6f} [MHz]")
print(f"PTFE (F) - Guía:      a = {a2:.6f} ± {sigma_a2:.6f}  [MHz/mT],  c = {c2:.6f} ± {sigma_c2:.6f} [MHz]")

# ===============================
# LÍNEAS SUAVES PARA PLOT
# ===============================
B0_1_fit = np.linspace(B0_1.min(), B0_1.max(), 200)
B0_2_fit = np.linspace(B0_2.min(), B0_2.max(), 200)

f_1_fit = a1 * B0_1_fit + c1
f_2_fit = a2 * B0_2_fit + c2

# ===============================
# LIMITES DE EJES
# ===============================
xmin = min(B0_1.min(), B0_2.min()) - 5
xmax = max(B0_1.max(), B0_2.max()) + 5
ymin = min(f_1.min(), f_2.min()) - 0.3
ymax = max(f_1.max(), f_2.max()) + 0.3

# ===============================
# ESTILO OSCURO + FORMAS DISTINTAS
# ===============================
color_H = 'navy'         # azul oscuro
color_F = 'darkgreen'    # verde oscuro

marker_H = 'o'           # círculo
marker_F = 's'           # cuadrado

line_H = '-'             # línea sólida
line_F = '-'            # línea discontinua

# ==========================================================
# FIGURA: ARRIBA f vs B0, ABAJO RESIDUALES
# ==========================================================
fig, (ax1p, ax2p) = plt.subplots(
    2, 1,
    sharex=True,
    figsize=(7.8, 7.2),
    dpi=160,
    gridspec_kw={'height_ratios': [3, 1]}
)

fig.patch.set_facecolor('white')
ax1p.set_facecolor('white')
ax2p.set_facecolor('white')

fig.suptitle("Resultados de la guía del experimento", fontsize=12)

# ---------- PANEL SUPERIOR ----------
ax1p.errorbar(B0_1, f_1, xerr=sigma_B0, yerr=sigma_f,
              fmt=marker_H, markersize=5, markeredgewidth=0.8,
              capsize=3, elinewidth=0.9, color=color_H)
ax1p.plot(B0_1_fit, f_1_fit, lw=1.4, linestyle=line_H, color=color_H)

ax1p.errorbar(B0_2, f_2, xerr=sigma_B0, yerr=sigma_f,
              fmt=marker_F, markersize=5, markeredgewidth=0.8,
              capsize=3, elinewidth=0.9, color=color_F)
ax1p.plot(B0_2_fit, f_2_fit, lw=1.4, linestyle=line_F, color=color_F)

ax1p.set_xlim(xmin, xmax)
ax1p.set_ylim(ymin, ymax)

ax1p.yaxis.set_major_locator(MultipleLocator(1.0))
ax1p.yaxis.set_minor_locator(MultipleLocator(0.1))
ax1p.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

ax1p.set_ylabel(r'$f\ \mathrm{[MHz]}$', fontsize=10)
ax1p.tick_params(axis='both', which='major', labelsize=8, length=6, width=0.9)
ax1p.tick_params(axis='both', which='minor', length=3, width=0.5)
ax1p.grid(False)

# ----- CAJAS DE TEXTO -----
texto_glicerina = (
    r"$\text{— Glicerina (H) [Guía]}$" "\n"
    rf"$f(B_0)=({a1:.3f})\,B_0 {c1:+.3f}$" "\n"
    rf"$\gamma=({a1:.3f}\pm{sigma_a1:.3f})\ \mathrm{{MHz/mT}}$"
)

ax1p.text(0.03, 0.97, texto_glicerina,
          transform=ax1p.transAxes, ha='left', va='top',
          multialignment='left', fontsize=8, color=color_H,
          bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                    edgecolor=color_H, linewidth=0.8))

texto_ptfe = (
    r"$\text{— PTFE (F) [Guía]}$" "\n"
    rf"$f(B_0)=({a2:.3f})\,B_0 {c2:+.3f}$" "\n"
    rf"$\gamma=({a2:.3f}\pm{sigma_a2:.3f})\ \mathrm{{MHz/mT}}$"
)

# AQUÍ lo bajamos: (derecha, parte inferior)
ax1p.text(0.97, 0.10, texto_ptfe,
          transform=ax1p.transAxes, ha='right', va='bottom',
          multialignment='left', fontsize=8, color=color_F,
          bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                    edgecolor=color_F, linewidth=0.8))

# ---------- PANEL INFERIOR: RESIDUALES ----------
ax2p.plot(B0_1, r1, marker_H, markersize=5, color=color_H)
ax2p.plot(B0_2, r2, marker_F, markersize=5, color=color_F)

ax2p.axhline(0, lw=0.9, color='black', alpha=0.7)
ax2p.set_xlim(xmin, xmax)

ax2p.xaxis.set_major_locator(MultipleLocator(10.0))
ax2p.xaxis.set_minor_locator(MultipleLocator(2.0))
ax2p.xaxis.set_major_formatter(FormatStrFormatter('%.0f'))

ax2p.set_ylabel(r'$\Delta f = f-\hat{f}\ \mathrm{[MHz]}$', fontsize=10)
ax2p.set_xlabel(r'$B_0\ \mathrm{[mT]}$', fontsize=10)

ax2p.tick_params(axis='both', which='major', labelsize=8, length=6, width=0.9)
ax2p.tick_params(axis='both', which='minor', length=3, width=0.5)
ax2p.grid(False)

ax2p.text(0.03, 0.88, "Glicerina (H) [Guía]", transform=ax2p.transAxes,
          ha='left', va='top', fontsize=8, color=color_H)
ax2p.text(0.97, 0.88, "PTFE (F) [Guía]", transform=ax2p.transAxes,
          ha='right', va='top', fontsize=8, color=color_F)

plt.tight_layout()
plt.subplots_adjust(top=0.92)

plt.savefig("figura_combinada.png", dpi=300, facecolor='white',
            edgecolor='white', bbox_inches='tight')
plt.show()

