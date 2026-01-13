<<<<<<< HEAD
import matplotlib as mpl
mpl.rcParams['animation.ffmpeg_path'] = r'C:\ffmpeg\bin\ffmpeg.exe'
# -*- coding: utf-8 -*-
"""
Xarxa neuromòrfica 'ADHD-like' amb Nengo per a Instagram
- Escenari A: Pantalles / novetat alta fins tard
- Escenari B: Ritual CBT-I minimalista (menys novetat, més control inhibidor)
Genera MP4 1080x1080 (1:1) o 1080x1350 (4:5) per Instagram.

Requisits:
  pip install nengo matplotlib numpy
  FFmpeg instal·lat (Matplotlib->FFMpegWriter)

Docs Nengo i LIF: https://www.nengo.ai/nengo/  (vegeu exemples de LIF) 
FFMpegWriter: Matplotlib docs
"""

import argparse
import numpy as np
import nengo
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from nengo.utils.matplotlib import rasterplot

# ---------- Paràmetres visuals Instagram ----------
# 1080x1080 (1:1) -> 10.8" x 10.8" a 100 dpi
# 1080x1350 (4:5) -> 10.8" x 13.5" a 100 dpi
PRESETS = {
    "square":  {"figsize": (10.8, 10.8), "dpi": 100, "tag": "1080x1080"},
    "portrait": {"figsize": (10.8, 13.5), "dpi": 100, "tag": "1080x1350"},
    "reel":    {"figsize": (10.8, 19.2), "dpi": 100, "tag": "1080x1920"}
}

# ---------- Perfils d’entrada ----------
def profile_circadian(T, dt, start=0.2, end=0.9):
    steps = int(T / dt)
    return np.linspace(start, end, steps)

def profile_novelty_A(T, dt):
    # Pantalles/novetat alta fins tard
    steps = int(T / dt)
    n = np.zeros(steps)
    ramp = int(0.66 * steps)
    n[:ramp] = np.linspace(0.4, 0.8, ramp)
    n[ramp:] = 0.7
    return n

def profile_novelty_B(T, dt):
    # Ritual: novetat baixa i estable a la nit
    steps = int(T / dt)
    n = np.zeros(steps)
    cut = int(0.60 * steps)
    n[:cut] = np.linspace(0.3, 0.1, cut)
    n[cut:] = 0.05
    return n

# ---------- Construcció del model ----------
def build_model(adhd_like=True, ritual=False, seed=0,
                N_e=80, N_i=32,
                w_ei=0.9, w_ie_base=0.45, w_ii=0.3, w_ee=0.2,
                boost_inhib=0.0,
                noise_e=0.15, noise_i=0.10,
                novelty_input_func=None, circ_input_func=None):
    """
    Xarxa E/I estil LIF amb nodes d'input per novetat i circadià.
    Si ritual=True, s'aplica un 'boost' inhibidor i menys soroll.
    """
    rng = np.random.RandomState(seed)
    model = nengo.Network(seed=seed, label="ADHD-like E/I")
    with model:
        # Nodes d'entrada (novetat i circadià)
        novelty = nengo.Node(novelty_input_func, size_in=0, size_out=1, label="novetat") if novelty_input_func else nengo.Node(size_in=0, size_out=1, label="novetat")
        circ = nengo.Node(circ_input_func, size_in=0, size_out=1, label="circadià") if circ_input_func else nengo.Node(size_in=0, size_out=1, label="circadià")

        # Inputs combinats com a corrents 'toniques'
        drive_e = nengo.Node(size_in=1, label="drive_E")
        drive_i = nengo.Node(size_in=1, label="drive_I")

        # Populacions LIF
        E = nengo.Ensemble(
            n_neurons=N_e, dimensions=1, label="Excitadores (E)",
            max_rates=nengo.dists.Uniform(80, 120),
            intercepts=nengo.dists.Uniform(-0.9, -0.4),
        )
        I = nengo.Ensemble(
            n_neurons=N_i, dimensions=1, label="Inhibidores (I)",
            max_rates=nengo.dists.Uniform(80, 120),
            intercepts=nengo.dists.Uniform(-0.9, -0.4),
        )

        # Connexions d'entrada (guanys diferents per E/I)
        # E rep més novetat/circadià; I una mica menys
        nengo.Connection(novelty, drive_e, transform=1.0, synapse=None)
        nengo.Connection(circ,    drive_e, transform=0.6, synapse=None)
        nengo.Connection(novelty, drive_i, transform=0.4, synapse=None)
        nengo.Connection(circ,    drive_i, transform=0.4, synapse=None)

        # Connecta drives a E/I
        nengo.Connection(drive_e, E, synapse=0.01)
        nengo.Connection(drive_i, I, synapse=0.01)

        # Recurrent E->E (excitació suau)
        nengo.Connection(E, E, transform=w_ee, synapse=0.05)
        # E->I (excita I)
        nengo.Connection(E, I, transform=w_ei, synapse=0.03)
        # I->E (inhibeix E) — aquí és on es veu “control inhibidor”
        w_ie = w_ie_base + (0.25 if ritual else 0.0) + boost_inhib
        nengo.Connection(I, E, transform=-abs(w_ie), synapse=0.03)
        # I->I (auto-inhibició suau per estabilitzar)
        nengo.Connection(I, I, transform=-abs(w_ii), synapse=0.03)

        # Soroll (novetat interna/dispersió); ritual el redueix
        ne = noise_e * (0.6 if ritual else 1.0)
        ni = noise_i * (0.7 if ritual else 1.0)
        noiseE = nengo.Node(nengo.processes.WhiteNoise(
            dist=nengo.dists.Gaussian(0, ne), seed=seed+1))
        noiseI = nengo.Node(nengo.processes.WhiteNoise(
            dist=nengo.dists.Gaussian(0, ni), seed=seed+2))
        nengo.Connection(noiseE, E, synapse=0.01)
        nengo.Connection(noiseI, I, synapse=0.01)

        # Probes: spikes i FR (filtrada)
        p_spk_E = nengo.Probe(E.neurons)
        p_spk_I = nengo.Probe(I.neurons)
        p_fr_E  = nengo.Probe(E, synapse=0.05)
        p_fr_I  = nengo.Probe(I, synapse=0.05)

    return model, {"novelty": novelty, "circ": circ}, {
        "p_spk_E": p_spk_E, "p_spk_I": p_spk_I, "p_fr_E": p_fr_E, "p_fr_I": p_fr_I
    }

# ---------- Simulació d’un escenari ----------
def run_scenario(label, T=6.0, dt=0.001, ritual=False, seed=0):
    # Perfils d'entrada
    circ = profile_circadian(T, dt, start=0.2, end=0.9)
    nov = profile_novelty_B(T, dt) if ritual else profile_novelty_A(T, dt)

    # Crea nodes d'entrada que retornen els valors dels perfils
    def novelty_input(t):
        idx = min(int(t / dt), len(nov) - 1)
        return nov[idx]

    def circ_input(t):
        idx = min(int(t / dt), len(circ) - 1)
        return circ[idx]

    # Construeix model amb les funcions d'entrada
    model, inputs, probes = build_model(ritual=ritual, seed=seed, novelty_input_func=novelty_input, circ_input_func=circ_input)

    with nengo.Simulator(model, dt=dt, seed=seed) as sim:
        sim.run(T)

    data = {
        "t": sim.trange(),
        "spkE": sim.data[probes["p_spk_E"]],
        "spkI": sim.data[probes["p_spk_I"]],
        "frE":  sim.data[probes["p_fr_E"]].ravel(),
        "frI":  sim.data[probes["p_fr_I"]].ravel(),
        "nov": nov, "circ": circ, "label": label
    }
    return data

# ---------- Animació ----------
def make_animation(dataA, dataB, out_path="nengo_adhd_instagram.mp4",
                   preset="square", fps=30, seconds=None, title="Cervell en mode nit"):
    st = PRESETS[preset]
    figsize, dpi = st["figsize"], st["dpi"]
    tag = st["tag"]

    # Durada
    if seconds is None:
        seconds = int(max(dataA["t"][-1], dataB["t"][-1]))
    total_frames = seconds * fps
    # Índex temporal per frame
    def idx_for_frame(f, data):
        tmax = min(data["t"][-1], f / fps)
        return np.searchsorted(data["t"], tmax)

    plt.rcParams.update({
        "font.size": 16,
        "axes.titlesize": 20,
        "axes.labelsize": 16
    })
    fig = plt.figure(figsize=figsize, dpi=dpi)
    gs = fig.add_gridspec(3, 2, height_ratios=[1.2, 1.2, 1.0], hspace=0.35, wspace=0.25)

    ax_fr_A = fig.add_subplot(gs[0, :])
    ax_fr_B = fig.add_subplot(gs[1, :])
    ax_raster = fig.add_subplot(gs[2, 0])
    ax_inputs = fig.add_subplot(gs[2, 1])

    # Línies FR
    line_frA, = ax_fr_A.plot([], [], color="#E74C3C", lw=3, label="FR Excitadora (A)")
    line_frB, = ax_fr_B.plot([], [], color="#27AE60", lw=3, label="FR Excitadora (B)")
    ax_fr_A.set_title("A) Pantalles i novetat alta (no baixa)")
    ax_fr_B.set_title("B) Ritual CBT‑I: menys novetat + més control")
    for ax in [ax_fr_A, ax_fr_B]:
        ax.set_xlim(0, max(dataA["t"][-1], dataB["t"][-1]))
        ax.set_ylim(0, max(dataA["frE"].max(), dataB["frE"].max()) * 1.2)
        ax.set_ylabel("FR (a.u.)")
        ax.grid(alpha=.2)

    # Raster (E) només escenari A per visual impact
    ax_raster.set_title("Spikes E (A) – el “soroll” de pantalles")
    ax_raster.set_xlabel("Temps (s)")
    ax_raster.set_ylabel("Neurona")
    ax_raster.set_xlim(0, dataA["t"][-1])

    # Inputs
    ax_inputs.set_title("Entrades: novetat i circadià")
    line_novA, = ax_inputs.plot([], [], color="#D35400", lw=2, label="Novetat (A)")
    line_novB, = ax_inputs.plot([], [], color="#16A085", lw=2, label="Novetat (B)")
    line_circ, = ax_inputs.plot([], [], color="#8E44AD", lw=2, label="Circadià")
    ax_inputs.set_xlim(0, max(dataA["t"][-1], dataB["t"][-1]))
    ax_inputs.set_ylim(0, 1.05)
    ax_inputs.legend(loc="upper left")
    ax_inputs.grid(alpha=.2)

    # Text overlay
    suptxt = fig.suptitle(f"{title} · Sortida MP4 {tag}", fontsize=22, fontweight="bold")

    def init():
        line_frA.set_data([], [])
        line_frB.set_data([], [])
        line_novA.set_data([], [])
        line_novB.set_data([], [])
        line_circ.set_data([], [])
        ax_raster.cla()
        ax_raster.set_title("Spikes E (A) – el “soroll” de pantalles")
        ax_raster.set_xlabel("Temps (s)")
        ax_raster.set_ylabel("Neurona")
        ax_raster.set_xlim(0, dataA["t"][-1])
        return (line_frA, line_frB, line_novA, line_novB, line_circ)

    def animate(f):
        iA = idx_for_frame(f, dataA)
        iB = idx_for_frame(f, dataB)

        # FR
        line_frA.set_data(dataA["t"][:iA], dataA["frE"][:iA])
        line_frB.set_data(dataB["t"][:iB], dataB["frE"][:iB])

        # Raster (E, A) fins al frame
        ax_raster.cla()
        rasterplot(dataA["t"][:iA], dataA["spkE"][:iA], ax=ax_raster, colors=["#E74C3C"])
        ax_raster.set_xlim(0, dataA["t"][-1])
        ax_raster.set_ylim(-1, min(60, dataA["spkE"].shape[1]))  # zoom visual
        ax_raster.set_xlabel("Temps (s)")
        ax_raster.set_ylabel("Neurona")
        ax_raster.set_title("Spikes E (A) – el “soroll” de pantalles")

        # Inputs
        line_novA.set_data(dataA["t"][:iA], dataA["nov"][:iA])
        line_novB.set_data(dataB["t"][:iB], dataB["nov"][:iB])
        line_circ.set_data(dataA["t"][:iA], dataA["circ"][:iA])
        return (line_frA, line_frB, line_novA, line_novB, line_circ)

    ani = animation.FuncAnimation(
        fig, animate, frames=total_frames, init_func=init, blit=False, interval=1000/fps
    )

    # Escriure MP4 (necessita FFmpeg)
    try:
        writer = animation.FFMpegWriter(fps=fps, bitrate=8000, metadata={
            "title": "ADHD vs Ritual – Nengo",
            "artist": "Ferran + M365 Copilot"
        })
        ani.save(out_path, writer=writer, dpi=dpi)
        plt.close(fig)
        print(f"[OK] Vídeo generat: {out_path}")
    except FileNotFoundError:
        print("[ERROR] FFmpeg no trobat. Instal·la FFmpeg per generar el vídeo MP4.")
        print("Pots descarregar FFmpeg de https://ffmpeg.org/download.html")
        print("O utilitza un altre writer com ImageMagick per GIF.")
        plt.close(fig)

# ---------- CLI ----------
def main():
    ap = argparse.ArgumentParser(description="ADHD-like Nengo network → Instagram MP4")
    ap.add_argument("--preset", choices=list(PRESETS.keys()), default="square",
                    help="Format Instagram: square (1080x1080), portrait (1080x1350) o reel (1080x1920)")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--seconds", type=int, default=30, help="Durada del vídeo")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    T = float(args.seconds)  # sim = video (1s -> 1s)
    print("[*] Simulant Escenari A (Pantalles)...")
    dataA = run_scenario("Pantalles", T=T, ritual=False, seed=args.seed)
    print("[*] Simulant Escenari B (Ritual CBT-I)...")
    dataB = run_scenario("Ritual", T=T, ritual=True, seed=args.seed)

    tag = PRESETS[args.preset]["tag"]
    out_path = args.out or f"nengo_ADHD_instagram_{tag}.mp4"
    print(f"[*] Renderitzant vídeo {tag}…")
    make_animation(dataA, dataB, out_path=out_path, preset=args.preset,
                   fps=args.fps, seconds=args.seconds,
                   title="Pantalles vs Ritual: activació neuronal al vespre")

if __name__ == "__main__":
    main()
=======
import matplotlib as mpl
mpl.rcParams['animation.ffmpeg_path'] = r'C:\ffmpeg\bin\ffmpeg.exe'
# -*- coding: utf-8 -*-
"""
Xarxa neuromòrfica 'ADHD-like' amb Nengo per a Instagram
- Escenari A: Pantalles / novetat alta fins tard
- Escenari B: Ritual CBT-I minimalista (menys novetat, més control inhibidor)
Genera MP4 1080x1080 (1:1) o 1080x1350 (4:5) per Instagram.

Requisits:
  pip install nengo matplotlib numpy
  FFmpeg instal·lat (Matplotlib->FFMpegWriter)

Docs Nengo i LIF: https://www.nengo.ai/nengo/  (vegeu exemples de LIF) 
FFMpegWriter: Matplotlib docs
"""

import argparse
import numpy as np
import nengo
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from nengo.utils.matplotlib import rasterplot

# ---------- Paràmetres visuals Instagram ----------
# 1080x1080 (1:1) -> 10.8" x 10.8" a 100 dpi
# 1080x1350 (4:5) -> 10.8" x 13.5" a 100 dpi
PRESETS = {
    "square":  {"figsize": (10.8, 10.8), "dpi": 100, "tag": "1080x1080"},
    "portrait": {"figsize": (10.8, 13.5), "dpi": 100, "tag": "1080x1350"},
    "reel":    {"figsize": (10.8, 19.2), "dpi": 100, "tag": "1080x1920"}
}

# ---------- Perfils d’entrada ----------
def profile_circadian(T, dt, start=0.2, end=0.9):
    steps = int(T / dt)
    return np.linspace(start, end, steps)

def profile_novelty_A(T, dt):
    # Pantalles/novetat alta fins tard
    steps = int(T / dt)
    n = np.zeros(steps)
    ramp = int(0.66 * steps)
    n[:ramp] = np.linspace(0.4, 0.8, ramp)
    n[ramp:] = 0.7
    return n

def profile_novelty_B(T, dt):
    # Ritual: novetat baixa i estable a la nit
    steps = int(T / dt)
    n = np.zeros(steps)
    cut = int(0.60 * steps)
    n[:cut] = np.linspace(0.3, 0.1, cut)
    n[cut:] = 0.05
    return n

# ---------- Construcció del model ----------
def build_model(adhd_like=True, ritual=False, seed=0,
                N_e=80, N_i=32,
                w_ei=0.9, w_ie_base=0.45, w_ii=0.3, w_ee=0.2,
                boost_inhib=0.0,
                noise_e=0.15, noise_i=0.10,
                novelty_input_func=None, circ_input_func=None):
    """
    Xarxa E/I estil LIF amb nodes d'input per novetat i circadià.
    Si ritual=True, s'aplica un 'boost' inhibidor i menys soroll.
    """
    rng = np.random.RandomState(seed)
    model = nengo.Network(seed=seed, label="ADHD-like E/I")
    with model:
        # Nodes d'entrada (novetat i circadià)
        novelty = nengo.Node(novelty_input_func, size_in=0, size_out=1, label="novetat") if novelty_input_func else nengo.Node(size_in=0, size_out=1, label="novetat")
        circ = nengo.Node(circ_input_func, size_in=0, size_out=1, label="circadià") if circ_input_func else nengo.Node(size_in=0, size_out=1, label="circadià")

        # Inputs combinats com a corrents 'toniques'
        drive_e = nengo.Node(size_in=1, label="drive_E")
        drive_i = nengo.Node(size_in=1, label="drive_I")

        # Populacions LIF
        E = nengo.Ensemble(
            n_neurons=N_e, dimensions=1, label="Excitadores (E)",
            max_rates=nengo.dists.Uniform(80, 120),
            intercepts=nengo.dists.Uniform(-0.9, -0.4),
        )
        I = nengo.Ensemble(
            n_neurons=N_i, dimensions=1, label="Inhibidores (I)",
            max_rates=nengo.dists.Uniform(80, 120),
            intercepts=nengo.dists.Uniform(-0.9, -0.4),
        )

        # Connexions d'entrada (guanys diferents per E/I)
        # E rep més novetat/circadià; I una mica menys
        nengo.Connection(novelty, drive_e, transform=1.0, synapse=None)
        nengo.Connection(circ,    drive_e, transform=0.6, synapse=None)
        nengo.Connection(novelty, drive_i, transform=0.4, synapse=None)
        nengo.Connection(circ,    drive_i, transform=0.4, synapse=None)

        # Connecta drives a E/I
        nengo.Connection(drive_e, E, synapse=0.01)
        nengo.Connection(drive_i, I, synapse=0.01)

        # Recurrent E->E (excitació suau)
        nengo.Connection(E, E, transform=w_ee, synapse=0.05)
        # E->I (excita I)
        nengo.Connection(E, I, transform=w_ei, synapse=0.03)
        # I->E (inhibeix E) — aquí és on es veu “control inhibidor”
        w_ie = w_ie_base + (0.25 if ritual else 0.0) + boost_inhib
        nengo.Connection(I, E, transform=-abs(w_ie), synapse=0.03)
        # I->I (auto-inhibició suau per estabilitzar)
        nengo.Connection(I, I, transform=-abs(w_ii), synapse=0.03)

        # Soroll (novetat interna/dispersió); ritual el redueix
        ne = noise_e * (0.6 if ritual else 1.0)
        ni = noise_i * (0.7 if ritual else 1.0)
        noiseE = nengo.Node(nengo.processes.WhiteNoise(
            dist=nengo.dists.Gaussian(0, ne), seed=seed+1))
        noiseI = nengo.Node(nengo.processes.WhiteNoise(
            dist=nengo.dists.Gaussian(0, ni), seed=seed+2))
        nengo.Connection(noiseE, E, synapse=0.01)
        nengo.Connection(noiseI, I, synapse=0.01)

        # Probes: spikes i FR (filtrada)
        p_spk_E = nengo.Probe(E.neurons)
        p_spk_I = nengo.Probe(I.neurons)
        p_fr_E  = nengo.Probe(E, synapse=0.05)
        p_fr_I  = nengo.Probe(I, synapse=0.05)

    return model, {"novelty": novelty, "circ": circ}, {
        "p_spk_E": p_spk_E, "p_spk_I": p_spk_I, "p_fr_E": p_fr_E, "p_fr_I": p_fr_I
    }

# ---------- Simulació d’un escenari ----------
def run_scenario(label, T=6.0, dt=0.001, ritual=False, seed=0):
    # Perfils d'entrada
    circ = profile_circadian(T, dt, start=0.2, end=0.9)
    nov = profile_novelty_B(T, dt) if ritual else profile_novelty_A(T, dt)

    # Crea nodes d'entrada que retornen els valors dels perfils
    def novelty_input(t):
        idx = min(int(t / dt), len(nov) - 1)
        return nov[idx]

    def circ_input(t):
        idx = min(int(t / dt), len(circ) - 1)
        return circ[idx]

    # Construeix model amb les funcions d'entrada
    model, inputs, probes = build_model(ritual=ritual, seed=seed, novelty_input_func=novelty_input, circ_input_func=circ_input)

    with nengo.Simulator(model, dt=dt, seed=seed) as sim:
        sim.run(T)

    data = {
        "t": sim.trange(),
        "spkE": sim.data[probes["p_spk_E"]],
        "spkI": sim.data[probes["p_spk_I"]],
        "frE":  sim.data[probes["p_fr_E"]].ravel(),
        "frI":  sim.data[probes["p_fr_I"]].ravel(),
        "nov": nov, "circ": circ, "label": label
    }
    return data

# ---------- Animació ----------
def make_animation(dataA, dataB, out_path="nengo_adhd_instagram.mp4",
                   preset="square", fps=30, seconds=None, title="Cervell en mode nit"):
    st = PRESETS[preset]
    figsize, dpi = st["figsize"], st["dpi"]
    tag = st["tag"]

    # Durada
    if seconds is None:
        seconds = int(max(dataA["t"][-1], dataB["t"][-1]))
    total_frames = seconds * fps
    # Índex temporal per frame
    def idx_for_frame(f, data):
        tmax = min(data["t"][-1], f / fps)
        return np.searchsorted(data["t"], tmax)

    plt.rcParams.update({
        "font.size": 16,
        "axes.titlesize": 20,
        "axes.labelsize": 16
    })
    fig = plt.figure(figsize=figsize, dpi=dpi)
    gs = fig.add_gridspec(3, 2, height_ratios=[1.2, 1.2, 1.0], hspace=0.35, wspace=0.25)

    ax_fr_A = fig.add_subplot(gs[0, :])
    ax_fr_B = fig.add_subplot(gs[1, :])
    ax_raster = fig.add_subplot(gs[2, 0])
    ax_inputs = fig.add_subplot(gs[2, 1])

    # Línies FR
    line_frA, = ax_fr_A.plot([], [], color="#E74C3C", lw=3, label="FR Excitadora (A)")
    line_frB, = ax_fr_B.plot([], [], color="#27AE60", lw=3, label="FR Excitadora (B)")
    ax_fr_A.set_title("A) Pantalles i novetat alta (no baixa)")
    ax_fr_B.set_title("B) Ritual CBT‑I: menys novetat + més control")
    for ax in [ax_fr_A, ax_fr_B]:
        ax.set_xlim(0, max(dataA["t"][-1], dataB["t"][-1]))
        ax.set_ylim(0, max(dataA["frE"].max(), dataB["frE"].max()) * 1.2)
        ax.set_ylabel("FR (a.u.)")
        ax.grid(alpha=.2)

    # Raster (E) només escenari A per visual impact
    ax_raster.set_title("Spikes E (A) – el “soroll” de pantalles")
    ax_raster.set_xlabel("Temps (s)")
    ax_raster.set_ylabel("Neurona")
    ax_raster.set_xlim(0, dataA["t"][-1])

    # Inputs
    ax_inputs.set_title("Entrades: novetat i circadià")
    line_novA, = ax_inputs.plot([], [], color="#D35400", lw=2, label="Novetat (A)")
    line_novB, = ax_inputs.plot([], [], color="#16A085", lw=2, label="Novetat (B)")
    line_circ, = ax_inputs.plot([], [], color="#8E44AD", lw=2, label="Circadià")
    ax_inputs.set_xlim(0, max(dataA["t"][-1], dataB["t"][-1]))
    ax_inputs.set_ylim(0, 1.05)
    ax_inputs.legend(loc="upper left")
    ax_inputs.grid(alpha=.2)

    # Text overlay
    suptxt = fig.suptitle(f"{title} · Sortida MP4 {tag}", fontsize=22, fontweight="bold")

    def init():
        line_frA.set_data([], [])
        line_frB.set_data([], [])
        line_novA.set_data([], [])
        line_novB.set_data([], [])
        line_circ.set_data([], [])
        ax_raster.cla()
        ax_raster.set_title("Spikes E (A) – el “soroll” de pantalles")
        ax_raster.set_xlabel("Temps (s)")
        ax_raster.set_ylabel("Neurona")
        ax_raster.set_xlim(0, dataA["t"][-1])
        return (line_frA, line_frB, line_novA, line_novB, line_circ)

    def animate(f):
        iA = idx_for_frame(f, dataA)
        iB = idx_for_frame(f, dataB)

        # FR
        line_frA.set_data(dataA["t"][:iA], dataA["frE"][:iA])
        line_frB.set_data(dataB["t"][:iB], dataB["frE"][:iB])

        # Raster (E, A) fins al frame
        ax_raster.cla()
        rasterplot(dataA["t"][:iA], dataA["spkE"][:iA], ax=ax_raster, colors=["#E74C3C"])
        ax_raster.set_xlim(0, dataA["t"][-1])
        ax_raster.set_ylim(-1, min(60, dataA["spkE"].shape[1]))  # zoom visual
        ax_raster.set_xlabel("Temps (s)")
        ax_raster.set_ylabel("Neurona")
        ax_raster.set_title("Spikes E (A) – el “soroll” de pantalles")

        # Inputs
        line_novA.set_data(dataA["t"][:iA], dataA["nov"][:iA])
        line_novB.set_data(dataB["t"][:iB], dataB["nov"][:iB])
        line_circ.set_data(dataA["t"][:iA], dataA["circ"][:iA])
        return (line_frA, line_frB, line_novA, line_novB, line_circ)

    ani = animation.FuncAnimation(
        fig, animate, frames=total_frames, init_func=init, blit=False, interval=1000/fps
    )

    # Escriure MP4 (necessita FFmpeg)
    try:
        writer = animation.FFMpegWriter(fps=fps, bitrate=8000, metadata={
            "title": "ADHD vs Ritual – Nengo",
            "artist": "Ferran + M365 Copilot"
        })
        ani.save(out_path, writer=writer, dpi=dpi)
        plt.close(fig)
        print(f"[OK] Vídeo generat: {out_path}")
    except FileNotFoundError:
        print("[ERROR] FFmpeg no trobat. Instal·la FFmpeg per generar el vídeo MP4.")
        print("Pots descarregar FFmpeg de https://ffmpeg.org/download.html")
        print("O utilitza un altre writer com ImageMagick per GIF.")
        plt.close(fig)

# ---------- CLI ----------
def main():
    ap = argparse.ArgumentParser(description="ADHD-like Nengo network → Instagram MP4")
    ap.add_argument("--preset", choices=list(PRESETS.keys()), default="square",
                    help="Format Instagram: square (1080x1080), portrait (1080x1350) o reel (1080x1920)")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--seconds", type=int, default=30, help="Durada del vídeo")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    T = float(args.seconds)  # sim = video (1s -> 1s)
    print("[*] Simulant Escenari A (Pantalles)...")
    dataA = run_scenario("Pantalles", T=T, ritual=False, seed=args.seed)
    print("[*] Simulant Escenari B (Ritual CBT-I)...")
    dataB = run_scenario("Ritual", T=T, ritual=True, seed=args.seed)

    tag = PRESETS[args.preset]["tag"]
    out_path = args.out or f"nengo_ADHD_instagram_{tag}.mp4"
    print(f"[*] Renderitzant vídeo {tag}…")
    make_animation(dataA, dataB, out_path=out_path, preset=args.preset,
                   fps=args.fps, seconds=args.seconds,
                   title="Pantalles vs Ritual: activació neuronal al vespre")

if __name__ == "__main__":
    main()
>>>>>>> 55a4ca1b4039dfdc32b27f650084aab22951f9c9
