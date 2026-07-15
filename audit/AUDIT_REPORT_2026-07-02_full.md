# Vollaudit 2026-07-02 — Chess AI (RTX 5070, 12 GB, SM_120)

Drittes Vollaudit (nach 2026-06-10/11 und 2026-06-12). Gelesen: gesamte Trainings-Pipeline
(main, trainer, self_play, replay_buffer, mcts/tree, model/*, engine/* inkl. C++),
alle Utils, alle Tools, alle Tests, CI, beide Vorgänger-Audits, uncommitted Diff,
Live-Zustand (checkpoints/, pool_state.json, supervisor.log). Vendored Stockfish
(tools/stockfish/) ausgenommen.

---

## EXECUTIVE SUMMARY

1. **Die Pipeline ist nach zwei Fix-Pässen mathematisch weitgehend korrekt.** PUCT
   (Vorzeichen, Formel), Dirichlet (echte Gamma-Samples, nur Root), Backup
   (alternierendes Vorzeichen), Temperatur (Log-Space), kanonische Encodierung
   (Board+Aktionsraum, bijektiv, getestet), WDL-Cross-Entropy + TD-Blend,
   BF16-Autocast mit fp32-Master-Weights, atomare Checkpoints mit RNG-State —
   alles verifiziert korrekt und durch Regressionstests gepinnt.
2. **Der eigentliche Befund ist strategisch, nicht technisch: die interne Elo-Kurve
   ist über 100 000 Trainingsschritte FLACH** (pool_state.json: 1000.0 → 1018.6).
   Die Arenen sind draw-dominiert (93 % Remis beobachtet), die Selbstverbesserungs-
   schleife dreht praktisch leer. Eine einzelne RTX 5070 liefert ~2–8 M Self-Play-
   Positionen/Tag; AlphaZero brauchte ~44 M *Partien*. Der bereits begonnene
   Distillations-Pivot (Stockfish/Puzzles → Teacher-Buffer) ist der richtige Weg
   und sollte zum Primärpfad werden.
3. **Ein aktiver nativer Crash (Access Violation 0xC0000005 in `RoundDriver.select`,
   tree.py:1153) besteht TROTZ der Serialisierung `cpp_round_threads=1` fort**
   (Supervisor-Log 06-27, 06-28 — nach Commit 02f76c1). Die Race-Hypothese ist
   damit widerlegt/unvollständig; es ist echte Heap-Korruption. Der Supervisor
   begrenzt den Schaden, aber der Fix kostet euch derzeit die Multithread-Selection
   ohne Gegenwert.
4. **Konkrete kleine Bugs gefunden:** (a) Teacher-Batch trainiert NUR die Policy —
   die Stockfish-WDL-Labels (bestes Value-Signal im System) werden weggeworfen;
   (b) das deferred PER-Priority-Update ist nur durch totes Code-Nebenprodukt
   race-frei; (c) In-Training-Arena nutzt den langsamen Single-Leaf-Pfad;
   (d) alle Tools (web_gui, run_arena, elo_vs_stockfish, bench_infer) laufen ohne
   TF32/cudnn.benchmark; (e) pyproject deklariert das falsche Paket
   (`python-chess` statt `chess`).
5. **GPU-seitig ist wenig Luft im Trainingsschritt** (2,6/12 GB Peak, fused AdamW,
   BF16, CUDA-Graph-Inferenz, GPU-seitiger History-Cache). Der Hebel ist nicht
   „Trainer schneller machen", sondern **Positionen/Stunde**: supervidierte Daten
   trainieren mit ~15–30 k Pos/s, Self-Play erzeugt ~26–100 Pos/s — Faktor
   ~300–1000.

---

## PHASE 0 — INVENTAR

~21 400 Zeilen Eigencode (ohne vendored Stockfish). Kernmodule:

| Bereich | Dateien | LOC | Zustand |
|---|---|---|---|
| C++-Engine | chess_core.{h,cpp}, mcts_core.{h,cpp}, bindings.cpp | ~2 400 | Magic Bitboards, Zobrist, RoundDriver; Perft+Paritätstests grün |
| MCTS-Python | mcts/tree.py (1538), node.py, dca.py | ~1 800 | Batch-Treiber, CUDA-Graph-Anbindung, GPU-History-Cache |
| Training | trainer.py (1052), self_play.py, replay_buffer.py, pbt.py | ~2 200 | WDL-CE, PER/SumTree, Teacher-KL, OpponentPool |
| Modell | network.py, heads.py, attention.py, inference.py, net2net, onnx | ~1 200 | ResNet 10×256 + SE + GRU-Encoder, 23,2 M Param |
| Orchestrierung | main.py (1268), train_supervisor, overnight | ~1 800 | Phasen, Tests, Crash-Supervisor |
| Tools | web_gui (3264, großteils HTML/JS), elo_vs_stockfish, run_arena, seed_teacher, … | ~6 000 | separat laufende Prozesse |
| Tests | 10 Dateien | ~1 250 | Regressionstests für alle Alt-Audit-Bugs |

**Dependencies:** torch 2.11.0/cu128 (Blackwell sm_120 offiziell unterstützt ✓),
numpy 2.4.3 ✓, chess 1.11.2 ✓, pybind11 3.0.4 ✓, tensorboard 2.20 ✓.
torch.compile auf Windows korrekt deaktiviert (kein Triton).
**Bug (S11):** `pyproject.toml:30` deklariert `python-chess>=1.11` — dieses
PyPI-Paket endet bei 1.1.0 (steht sogar in requirements.txt als Kommentar);
`pip install .` installiert eine antike/inkompatible Bibliothek. → `chess>=1.11`.

**Architektur-Philosophie:** AlphaZero-Kern + pragmatische Erweiterungen
(WDL-Head à la Lc0, TD-Blend à la KataGo, Teacher-Distillation, DCA,
Endgame-Curriculum, Resign mit No-Resign-Quote, Q-Adjudication am Move-Cap).
**Scope-Realismus:** Als Engineering-Projekt exzellent; als reines
From-Scratch-AlphaZero auf einer 5070 um ~3 Größenordnungen compute-unterversorgt.
Die Hybridisierung (supervised Seeds + Self-Play-Refinement) ist die richtige
Konsequenz und sollte konsequenter priorisiert werden (→ Phase 5/7).

---

## PHASE 1 — ARCHITEKTUR

**Datenfluss:** main → (system_probe → CONFIG in-memory) → build_model →
SelfPlayWorker → ParallelMCTS.game_stream → [RoundDriver C++ select → NN-Infer
(CUDA-Graph) → C++ finish] → GameRecord → PositionRecords → PER-Buffer →
Trainer.train_step (70 % Replay + 30 % Teacher) → Checkpoints/TB → OpponentPool/Arena.

* **Trennung:** sauber. Board-Logik (engine/), Suche (mcts/), Netz (model/),
  Training (training/) klar getrennt; Tools benutzen nur die öffentliche API.
* **State:** globaler CONFIG-Singleton wird zur Laufzeit mutiert (system_probe,
  Phasen-Overrides, ERED schreibt dirichlet_eps zurück). Dokumentiert und
  beabsichtigt, aber die Rückgabe-/Restore-Disziplin in phase_selfplay ist
  fehleranfällig — akzeptabel für Single-Process, würde bei Multi-Process brechen.
* **Reproduzierbarkeit:** Python/NumPy/Torch/CUDA geseedet, RNG-States im
  Checkpoint. ABER (S9): der C++-RNG (mcts_core.cpp:39, thread_local mt19937_64
  mit random_device) ist **ungeseedet** — Move-Sampling und Dirichlet-Noise im
  C++-Backend sind pro Lauf nichtdeterministisch. Die Aussage „praktisch
  reproduzierbar" (config.py:444) stimmt so nicht.
* **Checkpointing:** vorbildlich (tmp+os.replace, RNG, Buffer-Snapshots, Retention,
  Emergency-Save, Supervisor-Resume nach mtime).
* **Logging:** TB + Konsole vollständig (Losses, LR, Elo, pos/s, VRAM).
  Lücke: `elo_history.csv` wird nur am (nie erreichten) sauberen Pipeline-Ende
  geschrieben (main.py:1263) — Datei ist 17 Bytes groß. → in
  `_run_arena_and_update` mitschreiben.
* **Fehlerbehandlung:** NaN-Guard jeden Step vor backward ✓, Policy-Logit-Sanitizing ✓,
  faulthandler ✓, game_stream-FATAL-Dump ✓, Fail-open des RoundDrivers ✓.
* **Fünf kritischste fehlende Tests:**
  1. Teacher-Batch-Pfad im Trainer (hätte S3 gefangen — der Pfad ist ungetestet).
  2. CUDA-Graph-Engine ≡ Eager-Parität (infer-Äquivalenz ist ungepinnt).
  3. Fixed-Batch-Loss-Regressionstest (fester synthetischer Batch → erwarteter Loss).
  4. Langlauf-Fuzz von game_stream mit Assertion `chess_ext.stale_move_count()==0`
     (Frühwarnung für S1).
  5. Trainer `_save_buffers`/`_load_buffers`-Roundtrip inkl. `_prio_mult`.

---

## PHASE 2 — MIKROEBENE: BEFUNDE

### 2.0 Verifiziert korrekt (gegen Paper/Referenz nachgerechnet)

* PUCT `-Q - VL + c·P·√N_parent/(1+N+VL)` (mcts_core.cpp:54-67, node.py:56-70) ✓
* Dirichlet: `gamma_distribution(alpha,1)` normalisiert, nur Root, ε live aus CONFIG ✓
* Backup alternierendes Vorzeichen, Virtual-Loss add/undo symmetrisch ✓
* Temperatur-Sampling log-space mit Max-Subtraktion (Overflow-fix) ✓
* Kanonische Encodierung: sq^56-Flip + Farbblock-Swap, identisch in C++/Python/
  Aktionsraum; Mirror-Orakel-Tests ✓; Underpromotions beide Farben ✓
* En-passant-Legalitäts-Gate (gepinnter Bauer setzt kein ep-Feld) ✓
* Threefold/50-Züge/Insufficient exakt python-chess-paritätisch ✓
* WDL-CE mit Soft-Targets + 0.5·MSE auf TD-geblendetes Ziel (fixes Search-Q,
  kein Self-Bootstrap) ✓; Aux-Head speist den Value-Tower ✓
* OneCycleLR mit per-Group max_lr ✓; fused AdamW mit Fallback ✓;
  Grad-Clip 5.0 ✓; zero_grad(set_to_none) ✓
* fp32-Master-Weights + Autocast; GRU als fp32-Insel mit explizit
  deaktiviertem Autocast (cuDNN-RNN-Fastpath) ✓
* Move-Cap-Adjudication über finales Root-Q mit Side-Korrektur ✓;
  WDL-Labels aus aufgezeichneter Seite (nicht mn%2) ✓

### 2.1 Neue Befunde (Datei:Zeile, Schwere)

**S1 — KRITISCH (Stabilität): Access Violation in `RoundDriver.select` besteht
trotz `cpp_round_threads=1` fort.**
Beleg: runs/supervisor.log Crashes 2026-06-27 15:59 und 2026-06-28 01:04
(exit=3221225477) mit Stack in tree.py:1153 → `driver.select(...)` — beide NACH
Commit 02f76c1 (06-22), der select serialisiert. Mit `n_threads=1` läuft
`parallel_for` inline (mcts_core.cpp:414) — der Crash ist also single-threaded.
Konsequenzen: (a) die Race-Hypothese im config-Kommentar (config.py:193-200) ist
widerlegt, (b) ihr zahlt den Durchsatzverlust der Serialisierung ohne Gegenwert.
Verdachtsliste: `py::nodelete`-Node-Pointer, die `advance_root`-Freigaben
überleben (bindings.cpp:143); Double-Free-Interaktion `_needs_cpp_reset` ↔
`advance_root`; Folgeschaden eines Stale-Move (Guard chess_core.cpp:859 zählt —
Zähler wird aber nirgends periodisch geloggt).
Debug-Plan: (1) `chess_ext.stale_move_count()` pro Self-Play-Batch loggen
(10 min); (2) MSVC-ASan-Build (`/fsanitize=address /Zi` in setup_ext.py) für eine
Repro-Session, alternativ WSL-Build mit `-fsanitize=address`; (3) bis zur Klärung
threads=1 belassen, danach auf auto (≈4) zurück.

**S2 — HOCH (RL): Interne Elo flach; Arena diskriminiert nicht.**
pool_state.json: net_step0=1000.0 → net_step100023=1018.6. Ursachenkette:
Arena-Partien (greedy, 128 Sims, Cap 300) enden überwiegend remis, weil beide
Netze Gewinnstellungen nicht konvertieren → Elo-Updates ≈ 0 → Phasen-Gates
(target_elo 1600) unerreichbar → Phasenlogik läuft nur über max_games.
Fixes: (a) Arena-Partien am Cap wie im Self-Play über finales Q adjudizieren
(trainer.run_arena, trainer.py:1008: `result = r` bei Cap-Hit → stattdessen
Q-Adjudication analog `_finalize_record`); (b) objektive SF-Elo-Messung
(elo_vs_stockfish, batched) als Phasen-Gate statt Pool-Elo; (c) den Distillations-
pfad als Haupttreiber akzeptieren (→ Phase 7 Stufe 2.1).

**S3 — MITTEL (Korrektheit/verschenktes Signal): Teacher-Batch trainiert nur die
Policy; WDL-Labels der Teacher-Daten werden verworfen.**
training/trainer.py:284 (`policy_logits_t, wdl_t_pred, _ = self.model(...)`) und
:354-360 — `teacher_loss` ist nur KL; `wdl_t_t` (:270) wird transferiert und nie
benutzt. 30 % jedes Batches durchlaufen den Value-Head umsonst, und die
dezisiven Stockfish-Ergebnisse (das sauberste Value-Signal im System) erreichen
den Value-Head nie. Fix (Drop-in):
```python
if teacher_records:
    log_probs_t  = F.log_softmax(policy_logits_t.float(), dim=-1)
    teacher_loss = F.kl_div(log_probs_t, pol_t_t, reduction="batchmean")
    # NEU: Value-CE auf die Teacher-Outcomes (gleiche Soft-Target-Konstruktion)
    wdl_logits_t = wdl_t_pred.float()
    p_w = (2.0 * wdl_t_t - 1.0).clamp(min=0.0)
    p_l = (1.0 - 2.0 * wdl_t_t).clamp(min=0.0)
    p_d = (1.0 - p_w - p_l).clamp(min=0.0)
    tgt = torch.stack([p_w, p_d, p_l], dim=1)
    value_ce_t = -(tgt * F.log_softmax(wdl_logits_t, dim=-1)).sum(-1).mean()
    loss = loss + self.cfg.teacher_loss_weight * (teacher_loss + 0.5 * value_ce_t)
else: ...
```
Aufwand 30 min; erwarteter Effekt: deutlich kalibrierterer Value-Head (der
flache Value-Loss war historisch euer Kernproblem).

**S4 — MITTEL (fragile Korrektheit): Deferred-PER-Update ist nur zufällig
race-frei; gleichzeitig totes, teures Code-Stück.**
trainer.py:404-406 enqueued einen non-blocking D2H-Copy in pinned Memory; :214-217
liest ihn im nächsten Step OHNE Event-Sync. Einziger Grund, warum das heute
funktioniert: trainer.py:431-433 berechnet jeden Step eine Policy-Entropie `_ent`
(voller Softmax über B×4672 + `.item()`-Sync) — **und `_ent` wird nie benutzt**
(auf Log-Steps wird `ent` separat neu berechnet, :450-452). Entfernt jemand das
tote Stück, wird das PER-Update zu echtem Garbage-Read. Fix:
```python
td_pinned.copy_(td_err_gpu, non_blocking=True)
self._td_ev = torch.cuda.Event(); self._td_ev.record()
self._deferred_td_update = (indices, td_pinned)
# beim Konsum:
self._td_ev.synchronize()
self.replay_buffer.update_priorities(prev_idx, prev_td.numpy())
```
…und Zeilen 429-433 (`_ent`) ersatzlos streichen. Aufwand 15 min; spart zudem
einen Voll-Sync + Softmax pro Step.

**S5 — MITTEL (Perf): Alle Tools laufen ohne TF32/cudnn.benchmark.**
`torch.backends.cuda.matmul.allow_tf32` / `cudnn.allow_tf32` / `cudnn.benchmark` /
`set_float32_matmul_precision` werden NUR in main.py:171-181 gesetzt. web_gui,
run_arena, elo_vs_stockfish, bench_infer, uci_engine, lichess_bot laufen ohne —
d. h. fp32-GRU-Matmuls ohne TF32 und Convs ohne Autotune. **bench_infer misst
dadurch systematisch zu langsam.** Fix: Backend-Setup in eine Funktion (z. B. am
Anfang von `build_model`) ziehen. 15 min, +30–100 % in den Tools.

**S6 — MITTEL (Perf): In-Training-Arena nutzt den Single-Leaf-Pfad.**
main.py:461 `trainer.run_arena(model, opp_model, 100)` — der neue `batched_bs`-
Parameter (uncommitted Diff) wird nur von tools/run_arena.py genutzt. Die alle
10 k Steps laufende Arena feuert ~100 Partien × ~128 einzelne GPU-Forwards pro
Zug. Fix: `trainer.run_arena(model, opp_model, CONFIG.arena_games, batched_bs=32)`
(nebenbei: `100` ignoriert `config.arena_games=200` — toter Knopf). 5 min,
Arena ~10–20× schneller.

**S7 — NIEDRIG (Perf): PER-Sampling/Update sind reine Python-Loops.**
replay_buffer.py:284-291 (512 Baum-Descents) und :306-310. Bei ~60 Steps/Batch
einige ms/Step. Vektorisierbar (Batch-Descent über NumPy-Arrays), aber gemessen
am GPU-Anteil nachrangig.

**S8 — MITTEL (RL): Statischer 150 k-Teacher-Set mit konstantem Gewicht 1.0.**
154 Teacher-Samples/Step ⇒ nach 100 k Steps ist jede Teacher-Position ~100×
gesehen. Risiko: Policy memorisiert Skill-4/8/12-Stockfish-Züge und deckelt.
Empfehlung: `teacher_loss_weight` über ~50 k Steps 1.0 → 0.3 annealen; Teacher-Set
periodisch mit stärkerem SF (Skill 12–16, mehr movetime) neu seeden;
`loss/teacher_kl` beobachten — Drift → 0 = Memorisierung.

**S9 — NIEDRIG (Repro):** C++-RNG ungeseedet (s. Phase 1).

**S10 — MITTEL (Ops): Projekt liegt in OneDrive.**
278 MB-Checkpoints alle ~20 min + replay.pkl.gz (95 MB) + TB-Events → Dauersync,
CPU/Disk-Overhead, und OneDrive-Locks sind eine klassische Ursache für
`os.replace`-Fehler unter Windows. → checkpoints/, runs/, replay_buffer/ vom
Sync ausschließen („Immer behalten auf diesem Gerät" + Ordner-Ausschluss) oder
Repo aus OneDrive heraus verschieben.

**S11 — NIEDRIG:** pyproject-Paketname (s. Phase 0).

**S12 — INFO (Perf, C++):** `legal_moves()` prüft Legalität per Copy-Make
(chess_core.cpp:632-639) inkl. Kopie des `hash_history`-Vectors pro Pseudo-Move.
Gemessen 1,6 µs/Call — bei NN-gebundenem Self-Play irrelevant; nur relevant,
falls Select je dominiert (dann make/unmake).

**S13 — MITTEL (Architektur-Experiment): GRU-History-Encoder.**
Kosten: fp32-Insel (~12 % Forward-Latenz), History = ~8/9 des H2D-Volumens im
Trainer, historischer Bug-Magnet, und die Teacher-Daten haben durchweg
Null-Historien (Distributionsbruch für 30 % jedes Batches!). Der A/B-Schalter
(`use_gru_history=False`) existiert seit dem 06-12-Audit — **der Versuch wurde
nie gefahren.** AZ/Lc0 nutzen gestapelte History-Planes statt RNN. Empfehlung:
beim nächsten Fresh-Run (oder Distillations-Neustart) GRU-los starten.

**S14 — NIEDRIG (Hygiene):** Tote Knöpfe/Strukturen: `dataloader_workers`,
`prefetch_factor`, `persistent_workers`, `pin_memory` (kein DataLoader-Pfad),
`ReplayDataset`/`collate_records` (Trainer sampelt direkt),
`SelfPlayWorker.run_single_game` (Produktion nutzt game_stream), `aux_steps`
(deklariert legacy). Entfernen oder als „reserviert" markieren.

**S15 — NIEDRIG (I/O): Synchroner Buffer-Dump blockiert Training.**
`_save_buffers` (trainer.py:672-715) picklet+gzippt ~8 GB RAM-Daten zu 95 MB —
grob 30–90 s, alle 200 Steps (~20 min) ⇒ ~3–7 % Wall-Time, synchron im
Trainingspfad. Fix: Dump in Hintergrund-Thread (Snapshot von `_ptr/_size` +
Referenzliste reicht; Records sind nach dem Schreiben immutable) oder
`buffer_save_every=2–3`.

### 2.2–2.5 Kurzurteile

* **Schachlogik:** vollständig (EP inkl. Pin-Gate, Rochade-Rechte-Masken,
  4 Promotions, Threefold via Zobrist-Historie, 50-Züge, Stalemate/Mate,
  Insufficient-Material python-chess-paritätisch). Perft + 200-Spiele-Differential
  grün. Keine neuen Befunde.
* **Move-Encoding:** bijektiv (Test), kanonischer Flip konsistent Board↔Policy,
  C++↔Python bit-gleich (Test). ✓
* **Self-Play:** Turn-Handling ✓, Perspektiven ✓, Slot-Recycling hält Batch voll ✓.
  Data-Augmentation über Spiegelungen ist durch die kanonische Encodierung
  obsolet (Farbsymmetrie ist eingebaut); horizontale Spiegelung wäre wegen
  Rochade-Asymmetrie schachlich falsch — korrekt weggelassen.
* **Speicher/Nebenläufigkeit:** Ring-Puffer sauber; Single-Process-Design
  vermeidet Sync-Probleme; `start_background` trägt eine korrekte Warnung.
  Verbleibendes Risiko ist S1 (nativ).

---

## PHASE 3 — GPU (RTX 5070, 12 GB, Blackwell)

### 3.1 VRAM-Budget
23,2 M Param fp32 = 93 MB; AdamW-Momente 186 MB; Grads 93 MB ⇒ ~0,4 GB statisch.
Aktivierungen bei B=1024, 10 Blöcke, BF16: ~0,7–1,2 GB. Gemessen: 2,6 GB Peak —
**VRAM ist massiv unterausgelastet, und das ist OK**: der Trainingsschritt ist
nicht der Engpass. Batch NICHT blind erhöhen (bringt nichts, s. u.).
Gradient-Checkpointing bei ≤14 Blöcken korrekt deaktiviert.

### 3.2 Trainingsgeschwindigkeit
* AMP: BF16-Autocast ohne Scaler ✓ (richtig für Blackwell).
* TF32: gesetzt (nur main.py — S5 für Tools).
* cudnn.benchmark ✓, statische Shapes ✓.
* torch.compile: auf Windows korrekt aus; unter WSL2 realistisch +20–40 % auf den
  Forward — Stufe 4.
* CUDA-Graphs im Trainer: möglich, aber Aufwand/Nutzen schlecht (Trainer ≪
  Self-Play-Kosten).
* **Sync-Punkte im Hot-Loop:** (1) `loss_val = float(loss.detach().cpu())` — 
  bewusst (NaN-Guard), OK; (2) `_ent`-Softmax+`.item()` JEDEN Step — tot,
  streichen (S4); (3) Log-Cadence-Stack→CPU — OK (1 Sync/100 Steps);
  (4) deferred PER — Event-Sync nachrüsten (S4).
* DataLoader: existiert nicht (direktes Sampling) — bei 25 MB H2D/Step die
  richtige Entscheidung; `pin_memory`-Knöpfe sind tot (S14).

### 3.3 Self-Play-Inferenz (der echte Engpass)
* Batched über alle Bäume ✓ (ParallelMCTS + C++-RoundDriver, GIL-frei).
* CUDA-Graph-Engine mit fixer Batch-Breite = parallel_games (hw-derived 160) ✓;
  eval + inference_mode ✓; ONNX evaluiert und korrekt als optional eingestuft.
* GPU-History-Cache pro Zug (statt pro Runde) ✓ — der frühere 64 %-Transferanteil
  ist behoben.
* Rest-Ineffizienzen: (a) Graph rechnet immer volle Pool-Breite, auch wenn nur
  P<160 Leaves pending sind — bewusster Trade, ok; (b) statische Graph-Buffer
  sind fp32, Caller senden bf16 → doppelte Casts + größere Copies (klein);
  (c) `cpp_round_threads=1` (S1) serialisiert die CPU-Seite der Runde.
* Pipeline-Parallelisierung CPU↔GPU: wurde gebaut, gemessen (~0,6×) und
  begründet verworfen — richtige Entscheidung bei GPU-Bindung.

### 3.4 Netzarchitektur-Effizienz
* 256 ch / 21→25 Planes / 4672 Actions: Tensor-Core-freundlich (÷8/16) ✓.
* GroupNorm statt BatchNorm: richtig für die kleinen/variablen Batches der
  Inferenz.
* GELU: marginal teurer als ReLU, irrelevant gegen Convs.
* **Dimensionierung:** 23,2 M Parameter bei ~26–100 selbst erzeugten Pos/s sind
  viel Netz für wenig Daten. Für die aktuelle Datenrate wäre 128 ch × 10 Blöcke
  (~6 M, ~3–4× schnellere Inferenz = 3–4× mehr Self-Play-Daten) mutmaßlich
  Elo-pro-Woche-überlegen. Da Net2Net nur Tiefe wachsen lässt, wäre das ein
  Fresh-Run/Distillations-Neustart — als Experiment einplanen, nicht sofort.

### 3.5 Profiling-Anleitung (bitte Ergebnisse liefern)
```python
# einmalig in main.py hinter warmup_model() für 1 Batch aktivieren:
from torch.profiler import profile, ProfilerActivity, schedule
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
             schedule=schedule(wait=2, warmup=2, active=10),
             record_shapes=True, with_stack=False) as prof:
    for _ in range(14):
        trainer.train_step(); prof.step()
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=25))
```
Zusätzlich, während Self-Play läuft: `nvidia-smi dmon -s um -d 1` (60 s Fenster)
und `python tools/bench_infer.py 60` (NACH S5-Fix). Liefern: (1) die Profiler-
Tabelle, (2) mittlere GPU-Util % + VRAM aus dmon, (3) pos/s aus dem
Self-Play-Log bei 128 Sims. Damit lässt sich die Restlücke (Infer-Ceiling vs.
Produktion) exakt zuordnen.

---

## PHASE 4 — CPU/PYTHON

* Heißeste Python-Pfade sind nach den Fix-Pässen entschärft (move_to_action_index
  LRU-cached, Priors vektorisiert, Runde in C++). Verbleibend: PER-Loops (S7),
  dichte Policy-Rekonstruktion beim Batch-Bau (~6 MB/Step, ok), synchroner
  Buffer-Dump (S15).
* Multiprocessing: bewusst nicht (Single-GPU, GPU-gebunden) — richtig; Ray o. ä.
  wäre Overhead ohne zweite GPU.
* Serialisierung: pickle+gzip Level 3 — angemessen; nur die Synchronität stört (S15).

---

## PHASE 5 — RL-KORREKTHEIT & KONVERGENZ

* **Reward:** sauber (nur W/D/L; Q-Adjudication am Cap ist eine vertretbare,
  schwellen-gesicherte Heuristik; Kontempt im Training korrekt 0).
* **Explore/Exploit:** τ=1.0 für 30 Züge → 0.1, Dirichlet 0.3/0.25 mit
  ERED-Regelung, Resign 0.55/6 mit 15 % No-Resign, Endgame-Curriculum 12 % —
  durchdacht und für das Budget vernünftig.
* **Konvergenz-Realität:** Pool-Elo 1000→1018 über 100 k Steps (S2); externer
  Anker ~330 Elo vs. SF nach 8 h (Mitte Juni, Fresh-Run). Der 100 k-Kollaps
  (−158 Elo, Rollback auf 95 k, LR 5e-4) zeigt zusätzlich Fragilität bei langem
  Self-Play auf eigenem Datenstrom.
* **Erwartungsmanagement:** 1600/2200/2400-Phasenziele sind mit reinem Self-Play
  auf dieser Hardware nicht erreichbar. Erreichbar ist: supervidiert auf
  ~1800–2200 (SF-Distillation mit Value-Signal, Puzzles), danach Self-Play als
  Feinschliff — genau die Richtung, die ihr mit seed_teacher eingeschlagen habt.
  Konkret: `stockfish_distill` mit Skill 10–16, movetime 0.1–0.3, 10–30 k Partien
  (läuft CPU-seitig parallel zum Training!), Value-CE aktiv (S3), dann messen.
* **Evaluation:** elo_vs_stockfish ist nach den Fixes solide (Auto-Bisektion,
  faire Zeitbudgets, Skill-Anker). Empfehlung: alle ~5 k Steps automatisch 30
  Partien vs. festen SF-Anker (Skill 2–4) als objektive Kurve in TB loggen —
  die Pool-Elo ist als Fortschrittsmaß aktuell wertlos (S2).

---

## PHASE 6 — SYNTHESE

Die drei Perspektiven treffen sich in einem Punkt: **das System ist inzwischen
korrekt genug, dass Korrektheit nicht mehr der begrenzende Faktor ist — der
begrenzende Faktor ist Signalökonomie.**

* S2 (flache Elo) + S3 (Value-Signal der Teacher-Daten verworfen) + S8
  (statischer Teacher-Set) bilden eine Kette: Das beste verfügbare Lernsignal
  (dezisive SF-Partien) erreicht den Value-Head nicht, während der Policy-Head
  denselben 150 k-Set memorisiert und die draw-blinde Arena den (Nicht-)Fortschritt
  nicht einmal messen kann. Diese drei zusammen fixen = ein Nachmittag.
* S1 (AV-Crash) blockiert S-perf (Multithread-Select) und kostet über den
  Supervisor-Umweg Buffer-Rewinds; er ist der einzige echte Stabilitätsrest.
* S4/S5/S6/S15 sind unabhängige, kleine, risikofreie Gewinne.

---

## PHASE 7 — MASTERPLAN

**STUFE 1 — sofort (Korrektheit/Signal):**
1. S4: PER-Event-Sync + `_ent` löschen — trainer.py:404-406/429-433 — 15 min.
2. S3: Teacher-Value-CE — trainer.py:354-360 — 30 min. Erwartung: Value-Loss
   endlich mit echtem Signal; Kalibrierung des WDL-Heads.
3. S2a: Arena-Q-Adjudication + `arena_games` respektieren + batched_bs —
   trainer.py:1008 / main.py:461 — 45 min. Erwartung: Pool-Elo wird wieder
   ein Messinstrument; Arena 10–20× schneller.
4. S11: pyproject `chess>=1.11` — 2 min.
5. S1-Monitoring: `stale_move_count()`-Log pro Batch — 10 min.

**STUFE 2 — größter Speedup/Elo pro Aufwand:**
1. **Distillations-Offensive** (RL-Entscheidung, kein Code-Risiko): 10–30 k
   SF-Partien Skill 10–16 CPU-parallel harvesten, mit S3-Fix als sf_distill +
   seed_teacher einspielen. Erwartung: der mit Abstand größte Elo-Sprung pro
   GPU-Stunde (supervidiert trainiert ~300–1000× mehr Pos/h als Self-Play sie
   erzeugt).
2. S5: TF32/benchmark in Tools — 15 min — Tools +30–100 %.
3. S15: Buffer-Dump async/seltener — 30 min — +3–7 % Trainings-Wall-Time.
4. S8: teacher_loss_weight-Annealing 1.0→0.3 über 50 k — 15 min.
5. S1-Root-Cause (ASan-Session, 2–4 h) → danach `cpp_round_threads=0` (auto)
   reaktivieren — laut eurer eigenen Sweep-Messung +~10 % sims/s.

**STUFE 3 — mittel:**
6. Objektive SF-Elo-Kurve automatisiert (alle 5 k Steps, 30 Partien, TB-Log).
7. S13: GRU-Ablation (use_gru_history=False) als A/B beim nächsten Neustart —
   +10–15 % Durchsatz, weniger Komplexität, vermutlich kein Stärkeverlust.
8. S7: PER vektorisieren; S14: tote Knöpfe entfernen; elo_history.csv periodisch.
9. total_steps=500 000 → realistisch (~150 k) beim nächsten Fresh-Run, damit die
   Cosine-Decay-Phase real stattfindet.

**STUFE 4 — langfristig:**
10. WSL2-Betrieb → torch.compile/Triton (+20–40 % Forward), Linux-ASan.
11. Gumbel-AlphaZero-Selektion (Sequential Halving + Gumbel-Top-k): liefert bei
    16–64 Sims die Policy-Verbesserung, für die klassisches PUCT 128–800 braucht —
    das ist DER passende Algorithmus für Consumer-GPU-Budgets.
12. KataGo-Tricks: Playout-Cap-Randomization, Forced Playouts + Policy-Target-
    Pruning.
13. 128-ch-Netz-Experiment (Fresh-Run oder Distillation ins kleinere Netz).
14. C++ make/unmake statt Copy-Make (nur falls Select nach S1-Fix je dominiert).

---

## PHASE 8 — FINALES SETUP (RTX 5070)

**Config (Abweichungen vom Ist):**
```python
teacher_loss_weight   = 1.0   # → linear auf 0.3 bis Step 150k (Annealing einbauen)
arena_every_n_steps   = 10_000  # unverändert, aber Arena batched + adjudiziert
buffer_save_every     = 1     # behalten, sobald Dump async ist; sonst 3
cpp_round_threads     = 1     # bis S1 geklärt, danach 0 (auto)
use_gru_history       = False # beim NÄCHSTEN Fresh-Run (A/B)
total_steps           = 150_000  # nur bei Fresh-Run ändern (OneCycle-Resume!)
# unverändert richtig: bf16, cudagraph, batch 1024 (hw), pool 160 (hw),
# mcts_sims 128/256, lr 5e-4, warmup 2k, PER 0.6/0.4→1.0, draw-mult 0.5@25k
```

**Environment (bereits weitgehend live — zur Bestätigung):**
```python
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32      = True
torch.backends.cudnn.benchmark       = True
torch.set_float32_matmul_precision("high")
# → in EINE Funktion ziehen und aus build_model() aufrufen (S5)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:256")  # ✓ vorhanden
```

**Ops:** checkpoints/, runs/, replay_buffer/ aus OneDrive-Sync nehmen (S10).

**Mess-Schleife nach Umsetzung:** (1) tools/bench_infer.py 60 (nach S5),
(2) Profiler-Tabelle + dmon (Phase 3.5), (3) 30-Partien-SF-Anker vor/nach der
Distillations-Offensive, (4) stale_move_count-Verlauf. Diese vier Zahlen
entscheiden die nächste Iteration.
