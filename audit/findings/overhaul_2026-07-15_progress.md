# Überarbeitung 2026-07-15 — Umsetzungsstand (Plan: OVERHAUL_PROMPT_2026-07-15.md)

Alle Änderungen einzeln committet auf `feat/throughput-stability-webui`
(Commits `4c73d4c..5392616`, 25 Stück). Testsuite: 65 → **88 Tests, grün**.

## Block 0 — Baseline

Die uncommitteten Stufe-1/2-Fixes des 07-02-Audits (S3/S4/S5/S6/S7/S8/S11/S15)
in 8 logische Commits zerlegt und committet; Re-Audit-Checkliste in
`findings/reaudit_2026-07-15.md`.

## Block A — Training/Self-Play

| Punkt | Ergebnis |
|---|---|
| **A1 · S1-Crash** | **Instrumentiert, Root Cause noch offen.** ASan-Build-Option (`CHESS_AI_ASAN=1`) gebaut und verifiziert — aber ~90 sims/s (2.400× zu langsam für Volumen-Repro). Stattdessen: **Lifecycle-Canaries** in MCTSNode (Destruktor scribbelt DEAD; jeder Node-Zugriff prüft — best_child-Hotloop, expand, backup, mark_terminal) + **Stale-Root-Guards** (select_leaf/add_dirichlet_noise/RoundDriver.select verifizieren Root-Identität). Kosten ~0 (ein vorhersagbarer Compare neben ohnehin gelesenen Feldern). Repro-Harness `audit/scripts/stress_s1.py` (Stub-Engine, maximale native Churn): **45 min @ threads=1 (1,19 M Positionen) + 2 h @ threads=4, sims=96, cap=160 (527 k Positionen) — beide CLEAN**, stale=0, keine Canary-Auslösung. Konsequenz: künftige Vorkommen werden als präzise Exception am Berührungspunkt gemeldet (Fail-open in den Python-Loop) statt als Heap-Korruption Minuten später. `cpp_round_threads` bleibt 1 (Default); Re-Aktivierung auf 0 (auto, +~10 %) ist mit Supervisor + Tripwires vertretbar — Nutzerentscheidung. Nebenfund gefixt: `reset()` räumte `_needs_cpp_reset` nicht (erster Subtree jedes Folge-Spiels wurde weggeworfen). |
| **A2 · SF-Elo automatisch** | `utils/sf_anchor.py` + Gates in allen 5 Trainingsphasen: alle 5.000 Steps 30 Partien vs. festen SF-Anker → TB-Tags `elo/stockfish` + `elo/stockfish_score` + `checkpoints/sf_elo_history.csv`. Degenerierte Serien clampen auf ±400-Bound (`bound_only`). Anker-Default Skill 1 (~950), gewählt per Messung: Step-110k-Netz VERLOR 2 Schnellpartien vs. Skill 1 @ 32 Sims. Probe-Kosten ~5-10 min/5k Steps, `sf_elo_every_n_steps=0` schaltet ab. |
| **A3 · Engine-Harness** | `engine/harness.py`: load_engine (einheitliche Allowlist-Strictness — vorher DREI Policies), list_checkpoints (mit Pool-/SF-Elo-Labels), time_budget_s (clock/25+inc·¾), run_search (batched, unterbrechbar). Alle 6 Konsumenten migriert (uci, lichess, elo-Tool, web_gui, run_arena, main --play), je 1 Commit, verhaltensneutral verifiziert. |
| **A4 · GRU-Ablation** | Treiber `audit/scripts/gru_ablation.py` (Micro-Run verifiziert), **Overnight-Lauf gestartet** (4 h/Variante, Seeds fixiert via neuem set_rng_seed, + SF-Proben + 60-Partien-Direktvergleich). Schreibt `findings/gru_ablation_2026-07.md` selbst. Frühindiz aus dem Micro-Run: nogru ~+7 % pos/s. |
| **A5 · Self-Play vs. Distillation** | **Offen** (braucht die A2-Messlatte über mehrere Trainingsläufe). Werkzeuge stehen: SF-Elo-Kurve läuft in sf_distill/puzzles-Phasen mit; `sf_elo_history.csv` liefert Elo-pro-GPU-h direkt ablesbar. |
| **A6 · S9 RNG** | `chess_ext.set_rng_seed()` (Thread-local reseed + Seed-Stempel für spätere Threads); von main.py bei seed≠0 gerufen. Regressionstests: gleicher Seed ⇒ identische Noise+Samples. |
| **A7 · S14** | ReplayDataset, collate_records, dataloader_*-Knöpfe (config+system_probe+audit_network), run_single_game+_get_history, aux_steps entfernt — 0 Callsites (grep-belegt), Tombstone-Kommentare gesetzt. |
| **A8 · S10** | `CHESS_AI_DATA_DIR` verlagert checkpoints/+runs/ aus OneDrive; README-Abschnitt mit beiden Mitigationen. |

## Block B — Web-UI (3.291-Zeilen-Monolith → Paket)

| Punkt | Ergebnis |
|---|---|
| B1 | Frontend → `tools/webgui/static/` (index.html/style.css/app.js/app2.js, von Platte serviert, Traversal-Guard); Backend → `tools/webgui/{session,manager,monitor,server}.py` mit Routing-Tabelle; `web_gui.py` = 70-Zeilen-Launcher. Golden Path real im Browser verifiziert (Zug per Click-to-move mit Policy-Heatmap → KI-Antwort mit Kommentar → SSE-Suche/Tree-Viz → Takeback → Resign, 0 Konsolen-Fehler). |
| B2 | Multi-Session per **Tab-ID** (sessionStorage → `?sid=`, Cookie-Fallback), Cap 4 + LRU/Idle-Eviction, **ein** GPU-Lock serialisiert infer() inkl. Output-Klon (CUDA-Graph-Views!). Verifiziert: 2 Browser-Tabs spielten simultan e4/c6 und d4/Sf6 ohne State-Bluten. |
| B3 | Checkpoint-Dropdown (15 Einträge mit Step/Pool-Elo/SF-Elo-Labels), `/api/checkpoints` + `/api/engine` (Pfad-validiert), ModelCache (max 3, LRU). Beweis: Startpos-Policy-Differenz bis 0,32 nach Wechsel; altes Netz antwortet (e4→g5). **Stabilitätsfund:** Zweites Modell laden bei möglichem CUDA-Graph-Replay ⇒ `cudaErrorIllegalAddress` (Kontext vergiftet) — auch mit globalem GPU-Lock. Web-GUI-Prozess läuft deshalb komplett eager (~0,5 s/1500-Sim-Zug, Training/Self-Play behalten CUDA-Graph). |
| B4 | PGN-Persistenz (`runs/games/webgui_YYYYMM.pgn`), Settings-Persistenz (JSON, überlebt Neustart), 7 HTTP-Smoke-Tests (CPU, CI-fähig), ngrok-.bat mit Warnbanner + .gitignore-Sperre (die Datei enthält einen privaten Auth-Token!). |

## Block C — Spielen gegen die KI

| Punkt | Ergebnis |
|---|---|
| C1 | UCI auf batched Suche: **124 → ~8.300 nps (~66×)**, gemessen bei `go movetime 800`. |
| C2 | Echtes `go ponder`/`ponderhit` (Clock-Budget ab Hit, `bestmove ... ponder <zug>`), MultiPV 1-16, `UCI_LimitStrength`/`UCI_Elo` (Elo→Sims-Mapping, ~90 Elo/Verdopplung, geankert an der neuesten sf_elo_history-Messung). Skript-verifiziert (UCI_Elo 500 ⇒ 81-Knoten-Suche; ponderhit ⇒ bestmove nach exakt 1,2 s Budget). |
| C3 | Eine Zeitkontrolle (`time_budget_s`) + ein Suchkern (`run_search`) über UCI, Web, Terminal, Lichess-Bot (Bot war vorher clock-blind + single-leaf). |
| C4 | E2E: volle 38-Züge-Partie über das UCI-Protokoll mit echten Uhren (3+2, kein Flag-Fall, Matt-Ende, ~5,9k nps parallel zur Ablation); Web-GUI ausgiebig im Browser; Terminal-`--play` (e2e4→e7e5). Lichess-Bot code-migriert, Live-Test braucht Bot-Token-Session. |

## Offen / nächste Schritte

1. **S1 Root Cause**: Tripwires sind scharf; nächster Trainingslauf liefert
   entweder eine präzise Exception-Diagnose oder läuft durch. Optional:
   `cpp_round_threads=0` (auto) reaktivieren — Risikoprofil unverändert
   (Crash trat auch bei 1 auf), Diagnose jetzt deutlich besser.
2. **GRU-Ablation**: läuft (Ergebnis-Doc schreibt sich selbst nach ~9 h).
3. **A5-Messung**: nach dem nächsten sf_distill-/Self-Play-Zyklus die
   Elo/GPU-h beider Pfade aus `sf_elo_history.csv` ableiten und
   `teacher_game_ratio`/Reseeding/overnight.py-Allokation entscheiden.
4. Lichess-Bot einmal live gegen einen Menschen laufen lassen (Token nötig).
