# Chess AI

Eine selbstlernende Schach-KI nach AlphaZero-Prinzip: ein neuronales Netz wird
durch **Self-Play + MCTS** (Monte-Carlo Tree Search) ohne menschliche Partien
von Grund auf trainiert. Zuggenerierung und Suchbaum laufen in einer
C++-Erweiterung, das neuronale Netz in PyTorch (CUDA, BF16).

---

## Inhalt

- [Überblick](#überblick)
- [Architektur](#architektur)
- [Installation & Build](#installation--build)
- [Befehle](#befehle)
- [Trainingsablauf](#trainingsablauf)
- [Gegen die KI spielen](#gegen-die-ki-spielen)
- [ELO gegen Stockfish messen](#elo-gegen-stockfish-messen)
- [Monitoring](#monitoring)
- [Hardware-Autokonfiguration](#hardware-autokonfiguration)
- [Stabilität & Crash-Resistenz](#stabilität--crash-resistenz)
- [Realistische Performance](#realistische-performance)
- [Projektstruktur](#projektstruktur)

---

## Überblick

Der Trainingskreislauf:

```
   ┌─────────────────────────────────────────────────────────┐
   │  Self-Play: N Spiele parallel, MCTS + NN wählen Züge     │
   │  → Stellungen + Such-Policy + Ergebnis                   │
   └───────────────────────────┬─────────────────────────────┘
                               │  Positionen
                               ▼
   ┌─────────────────────────────────────────────────────────┐
   │  Replay-Buffer (Prioritized Experience Replay)           │
   │  + Teacher-Buffer (hochwertige 800-Sim-Partien)          │
   └───────────────────────────┬─────────────────────────────┘
                               │  Mini-Batches
                               ▼
   ┌─────────────────────────────────────────────────────────┐
   │  Trainer: Policy- + Value-Loss, TD(λ), Distillation      │
   │  → bessere Netz-Gewichte                                 │
   └───────────────────────────┬─────────────────────────────┘
                               │  neue Gewichte
                               └──────────► zurück zu Self-Play
```

Periodisch spielt das aktuelle Netz in einer **Arena** gegen ältere
Checkpoints; das ELO wird getrackt.

---

## Architektur

| Komponente | Datei | Beschreibung |
|---|---|---|
| C++-Engine | `chess_ext.*.pyd` | Zuggenerierung, Brett, MCTS-Baum, virtual loss |
| Netz | `model/network.py` | 10 Residual-Blöcke (256ch), SE-Attention, Policy- + Value-Kopf (~14 M Param.); per Net2Net auf 20 vergrößerbar |
| GRU-Encoder | `model/attention.py` | kodiert die letzten 8 Stellungen als Kontext (läuft fp32 wg. cuDNN) |
| MCTS | `mcts/tree.py` | `ParallelMCTS` — hält N Spiele gleichzeitig im Pool, ein gebatchter GPU-Forward pro Sim-Runde |
| Self-Play | `training/self_play.py` | erzeugt Partien, füllt die Buffer (persistenter Pool) |
| Trainer | `training/trainer.py` | Loss, Optimizer (AdamW + OneCycleLR), Checkpoints, Arena |
| Replay | `training/replay_buffer.py` | PER-Ringpuffer + Teacher-Buffer |
| Hardware-Probe | `utils/system_probe.py` | leitet Batch-Größe, Pool-Breite, Buffer-Größe aus GPU/CPU/RAM ab |

---

## Installation & Build

```bash
# 1. Python-Abhängigkeiten
pip install -r requirements.txt
# PyTorch mit CUDA separat von der offiziellen CUDA-Wheel-Quelle:
pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128

# 2. C++-Erweiterung bauen (im Projektordner chess_ai/)
python setup_ext.py build_ext --inplace
```

Nach dem Build muss `chess_ext.*.pyd` im `chess_ai/`-Ordner liegen.
Ohne die C++-Erweiterung fällt das Projekt auf `python-chess` zurück
(korrekt, aber deutlich langsamer).

---

## Befehle

Alle Befehle werden im Ordner `chess_ai/` ausgeführt.

### Integrationstests (immer zuerst ausführen)

```bash
python main.py --test
```

Prüft: Hardware-Probe, Perft (Zuggenerierung), C++/python-chess-Parität,
MCTS Matt-in-1, Forward-Pass/VRAM, 10 Trainingsschritte, Self-Play ohne
Deadlock, Checkpoint speichern/laden. **Muss komplett grün sein.**

### Training

```bash
python main.py --phase=all            # vollständige Pipeline (Standard)
python main.py --phase=pretraining    # nur Endspiel-Pretraining (braucht Syzygy)
python main.py --phase=selfplay       # nur Self-Play-Bootstrap (Phase 1)
python main.py --phase=distillation   # nur Teacher-Distillation (Phase 2)
python main.py --phase=refinement     # nur Feinschliff (Phase 3)
```

### Training fortsetzen

```bash
python main.py --phase=all --resume=checkpoints/step_0050000.pt
```

Lädt Netz, Optimizer, Scheduler **und** (falls vorhanden) die `.pkl`-Buffer
neben dem Checkpoint.

### Gegen die KI spielen

```bash
python main.py --play                           # du Weiß, neuester Checkpoint
python main.py --play --side black               # du Schwarz
python main.py --play --sims 400                 # stärkere KI (mehr MCTS-Sims)
python main.py --play --resume checkpoints/step_0050000.pt
```

### Syzygy-Tablebases (optional)

```bash
python main.py --phase=all --syzygy=C:\pfad\zu\syzygy
```

Aktiviert exakte Endspiel-Auswertung (≤5 Steine) und Endspiel-Pretraining.

### Alle Flags

| Flag | Werte | Standard | Bedeutung |
|---|---|---|---|
| `--phase` | `all`, `pretraining`, `selfplay`, `distillation`, `refinement` | `all` | Welche Trainingsphase(n) |
| `--resume` | Pfad zu `.pt` | – | Checkpoint laden und fortsetzen |
| `--test` | – | – | Nur Integrationstests, dann beenden |
| `--syzygy` | Ordnerpfad | – | Syzygy-Tablebase-Verzeichnis |
| `--play` | – | – | Interaktive Partie gegen die KI |
| `--side` | `white`, `black` | `white` | Deine Farbe bei `--play` |
| `--sims` | Ganzzahl | Config-Wert | MCTS-Sims pro KI-Zug bei `--play` |
| `--grow` | N | – | Net2Net: `--resume`-Checkpoint auf N Residual-Blöcke vergrößern, `*_grownN.pt` schreiben, beenden |

`Strg+C` während des Trainings speichert sauber einen Checkpoint vor dem
Beenden. Ein per `--resume` geladener Checkpoint mit abweichender Blockzahl
(z. B. nach `--grow`) passt die Architektur automatisch an.

### Zusatz-Tools (`tools/`)

```bash
python tools/verify_onnx.py                       # ONNX-Parität prüfen
python tools/bench_infer.py 60                    # Torch- vs ONNX-Durchsatz
python tools/elo_vs_stockfish.py --stockfish PFAD --games 30 --elo 1500
python tools/web_gui.py --sims 200                # Browser-Brett (http://127.0.0.1:8000)
python tools/lichess_bot.py                       # Bot auf lichess.org (LICHESS_TOKEN)
python tools/uci_engine.py                        # UCI-Engine für Schach-GUIs (Arena, Cute Chess, Nibbler …)
python main.py --play --show --ponder --sims 400  # Terminal: Züge/Eval sehen + Denken während du ziehst
python main.py --grow 20 --resume checkpoints/step_XXatXXXX.pt   # 10→20 Blöcke
```

`web_gui.py` und `lichess_bot.py` nutzen nur die Python-Standardbibliothek
(kein FastAPI/berserk). `verify_onnx.py`/`bench_infer.py` brauchen
`onnxruntime` (GPU-Beschleunigung nur mit `onnxruntime-gpu`).

---

## Trainingsablauf

`--phase=all` durchläuft nacheinander:

**Phase 0 — Endspiel-Pretraining** (nur mit `--syzygy`)
Supervidiertes Vortraining auf Tablebase-Endspielen. Ohne Syzygy übersprungen.

**Phase 1 — Self-Play-Bootstrap** (Ziel ~1600 ELO)
- *Fast-Bootstrap:* solange das Netz zufällig ist, nur 50 Sims/Zug
  (800 Sims auf einem Zufallsnetz bringen nichts, kosten aber 16×).
- N Spiele laufen kontinuierlich parallel; fertige Spiele werden sofort
  durch neue ersetzt, damit der GPU-Batch nie schrumpft.
- Ab Schritt 500 (echtes Trainingssignal vorhanden) werden die Sims auf
  200/800 hochgefahren.
- Training startet, sobald der Replay-Buffer ≥ 5 000 Stellungen hat.

**Phase 2 — Teacher-Distillation** (Ziel ~2200 ELO)
Volle Sim-Budgets, DCA aktiv (mehr Sims in kritischen Stellungen),
KL-Distillation gegen die hochwertigen Teacher-Partien.

**Phase 3 — Feinschliff** (Ziel ~2400 ELO)
Höherer Teacher-Anteil, Lernrate ×0,1.

Jede Phase endet bei Ziel-ELO **oder** Erreichen des Spiel-Limits.
Checkpoints landen in `checkpoints/step_*.pt`, ELO-Historie in
`checkpoints/elo_history.csv`.

---

## Gegen die KI spielen

```bash
python main.py --play
```

- Lädt automatisch den neuesten Checkpoint aus `checkpoints/`
  (ohne Checkpoint: untrainiertes Netz).
- Züge in **UCI-Notation** eingeben: `e2e4`, `g1f3`, Bauernumwandlung
  `e7e8q`.
- `quit` / `exit` / `q` beendet die Partie.
- Mehr `--sims` = stärkere, aber langsamere KI.

---

## ELO gegen Stockfish messen

Objektive Stärkemessung über UCI gegen Stockfish, der auf einen festen ELO
gepinnt wird (`UCI_LimitStrength` + `UCI_Elo`). Vor dem ersten Lauf einmal
Stockfish installieren (oder einen vorhandenen Pfad angeben):

```bash
python tools/install_stockfish.py
```

Stockfish landet unter `tools/stockfish/…`; der Pfad wird in
`runs/.stockfish_installed` gemerkt. Danach:

```bash
# Schnelltest: 20 Partien vs SF@1500, neuester Checkpoint
python tools/elo_vs_stockfish.py --games 20 --elo 1500

# Längerer Lauf mit explizitem Checkpoint + mehr Sims
python tools/elo_vs_stockfish.py --games 100 --elo 1800 \
    --sims 400 --movetime 0.2 \
    --resume checkpoints/step_0050000.pt
```

Ausgabe: W/D/L, ELO-Differenz zu SF und geschätztes Netz-ELO mit 95 %-CI.
Wenig Spiele → sehr breite CI (Faustregel: ≥ 100 Spiele für brauchbare Werte).

| Flag | Bedeutung |
|---|---|
| `--stockfish` | Pfad zur SF-Binary (Default: aus Installer-Sentinel) |
| `--games` | Anzahl Partien (Default 20, alternierende Farben) |
| `--elo` | SF-Stärke-Anker, 1320–3190 (Default 1500) |
| `--sims` | Eigene MCTS-Sims/Zug (Default: aus `config.py`) |
| `--movetime` | SF Bedenkzeit pro Zug in Sekunden (Default 0.1) |
| `--resume` | Checkpoint laden (Default: neueste `.pt` aus `checkpoints/`) |

---

## Monitoring

```bash
tensorboard --logdir=runs
```

Geloggt werden: Gesamt-/Policy-/Value-/Teacher-Loss, Lernrate, ELO,
Replay-Größe, Durchsatz (Positionen/s, Spiele/h), VRAM.

---

## Hardware-Autokonfiguration

Bei jedem Start prüft `utils/system_probe.py` GPU/CPU/RAM und leitet
**zur Laufzeit** (ohne `config.py` zu überschreiben) ab:

- **Präzision** (BF16, falls unterstützt)
- **Batch-Größe** (nach freiem VRAM)
- **Parallele Spiele** (nach GPU-Breite — Self-Play ist GPU-gebunden, nicht
  CPU-gebunden)
- **Replay-Buffer-Größe** (realistisch ~40 KB/Position effektiv inkl.
  Python-Overhead, gedeckelt auf 40 % des verfügbaren RAM → Ringpuffer
  recycelt, OS und MCTS-Pool behalten Headroom)
- **torch.compile** (auf Windows aus — kein Triton verfügbar)

Manuelle Defaults stehen in `config.py`.

---

## Stabilität & Crash-Resistenz

Das Training läuft oft mehrere Stunden bis Tage; daher sind mehrere
Verteidigungslinien gegen stille Abstürze eingebaut:

- **`faulthandler`** (in `main.py` aktiviert): native Crashes aus
  `chess_ext.pyd`, CUDA oder cuDNN landen mit Stacktrace im stderr-Log,
  statt den Prozess wortlos zu beenden.
- **`CUDA_LAUNCH_BLOCKING=1`** als Default: CUDA-Fehler werden synchron
  am Auslöser sichtbar. Vor dem produktiven Hochskalieren kann die
  Variable entfernt werden, sobald das Training stabil läuft.
- **Self-Play-Generator (`mcts/tree.py:game_stream`)** wrappt seinen
  Hauptloop in `try/except` und dumpt bei Crashs die aktuellen FENs
  aller Pool-Spiele sowie Zugnummern. Stille Generator-Tode (häufige
  Ursache von vermeintlich „spontanem Exit") sind so ausgeschlossen.
- **NaN/Inf-Schutz im Trainer**: vor jedem `loss.backward()` prüft der
  Trainer `torch.isfinite(loss)` und überspringt den Schritt sauber,
  statt AdamW-Momente mit NaN-Gradienten zu vergiften.
- **Policy-Logit-Clamp `[-30, 30]`** vor Softmax (`model/heads.py`):
  verhindert NaN-Lawinen aus runaway-Logits beim Bootstrap.
- **Replay-Buffer-Bounds**: Sparse-Policy-Indizes werden bei `__init__`
  und beim Reconstruct gegen die Aktionsgröße geprüft (`IndexError`
  statt stille Datenkorruption).
- **Vollständige Checkpoint-State**: `_entropy_history` und PER-Beta-Step
  (`replay_buffer._step`) wandern mit ins Checkpoint, so dass ein
  Resume die ERED-Kalibrierung und die IS-Gewichte nicht reißt.
- **Optimizer-Reset-Warnung**: falls ein Resume die AdamW-Momente nicht
  laden kann (Arch-Wechsel), wird das laut ins Log geschrieben — ein
  Loss-Spike danach ist erwartet, nicht ein Daten-Bug.
- **Strg+C** speichert sauber einen Checkpoint, bevor der Prozess endet.

Falls das Training trotzdem mal exit-t: im Log nach `[game_stream] FATAL`
oder einem `Fatal Python error:` vom faulthandler suchen — dort steht
der echte Auslöser, nicht erst die Folge-Symptome.

---

## Realistische Performance

Self-Play ist **GPU-gebunden** am Netz-Forward. Auf einer RTX 5070
(14-M-Netz, eager BF16):

- NN-Forward-Decke: ~10 000 Evals/s bei Batch 122
- ~280 k Positionen/h bei 50 Sims (Bootstrap)
- ~70 k Positionen/h bei 200 Sims (Standard)

> Die C++-Erweiterung beschleunigt **nur** Zuggenerierung/Suchbaum, **nicht**
> den GPU-Forward. Es gibt daher keinen großen „C++-Multiplikator".

Tempo-Hebel (alle GPU-seitig): kleineres Residual-Netz, weniger Sims,
oder `torch.compile` unter WSL/Linux (Triton dort verfügbar).
Ein CPU/GPU-Pipelining wurde getestet und war messbar **langsamer**
(Batch-Halbierung kostet mehr als das Overlap bringt) — bewusst nicht drin.

---

## Projektstruktur

```
chess_ai/
├── main.py                  Einstiegspunkt (Phasen, Tests, --play)
├── config.py                zentrale Konfiguration
├── setup_ext.py             Build-Skript der C++-Erweiterung
├── requirements.txt
├── chess_ext.*.pyd          gebaute C++-Engine
├── engine/                  Board, Zuggenerierung, Regeln (C++-Wrapper)
├── model/                   Netz, GRU-Encoder, Policy-/Value-Köpfe
├── mcts/                    MCTS-Baum, ParallelMCTS, DCA
├── training/                Self-Play, Trainer, Replay-Buffer, PBT
├── utils/                   Hardware-Probe, Logger, ELO
├── checkpoints/             gespeicherte Modelle + ELO-Historie
└── runs/                    TensorBoard-Logs
```

---

## Schnellstart

```bash
cd chess_ai
pip install -r requirements.txt
pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
python setup_ext.py build_ext --inplace
python main.py --test          # alles grün?
python main.py --phase=all     # Training starten
# … später:
python main.py --play          # gegen die KI spielen
```
