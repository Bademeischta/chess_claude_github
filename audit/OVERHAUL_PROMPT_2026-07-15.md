# Mission: Vollständige Analyse & Überarbeitung der Chess-AI (Fokus: Training/Self-Play, WebUI, Spielen gegen die KI)

Du übernimmst die Weiterentwicklung eines bestehenden AlphaZero-artigen Schach-Projekts (Python + C++-Extension, PyTorch, eigene MCTS-Suche). Das Projekt hat bereits eine sehr reife Selbst-Audit-Kultur (siehe `audit/AUDIT_REPORT_2026-06-12.md`, `audit/AUDIT_REPORT_2026-07-02_full.md`, `audit/findings/phase00-11_*.md`) — lies diese **zuerst vollständig**, bevor du irgendetwas änderst. Viele "offensichtlichen" Bugs sind dort bereits gefunden, teils gefixt, teils bewusst offen gelassen. Wiederhole keine Arbeit, die dort schon dokumentiert ist, sondern baue darauf auf.

Deine Aufgabe ist NICHT, ein neues Projekt zu bauen. Es ist, ein reifes, funktionierendes, aber an mehreren Stellen unfertiges/instabiles/unübersichtliches System auf professionelles Produktionsniveau zu heben — mit klarem Fokus auf drei Bereiche:

1. **Training-Loop & Self-Play** (Kern-RL-Pipeline: `training/`, `model/`, `mcts/`, `engine/`)
2. **Web-UI** (`tools/web_gui.py`, aktuell 3291 Zeilen Monolith)
3. **Spielen gegen die KI** (`tools/uci_engine.py`, `main.py --play`, `tools/web_gui.py` — alle drei sind aktuell unabhängige, sich duplizierende Pfade)

---

## Vorgehen — arbeite in dieser Reihenfolge

### Phase 0: Umgebungscheck (Pflicht, bevor du irgendetwas anfasst)
- Prüfe `git status`, `git log`, `git branch -a`. Es gibt möglicherweise einen parallelen Worktree/Branch (`claude/hopeful-mcnulty-1b2d74` bzw. `.claude/worktrees/...`) — kläre mit dem Nutzer, ob dort noch aktiv gearbeitet wird, bevor du große strukturelle Änderungen beginnst, um Merge-Konflikte oder doppelte Arbeit zu vermeiden.
- Prüfe, ob gerade ein Trainingslauf aktiv ist (`runs/supervisor.log`, `checkpoints/pool_state.json` Zeitstempel, laufende Prozesse). Das Repo liegt in OneDrive — sei vorsichtig mit Datei-Locks auf `checkpoints/*.pt` (bis zu 278MB) und dem Replay-Buffer während eines laufenden Trainings. Töte oder pausiere keinen laufenden Trainingsprozess ohne ausdrückliche Zustimmung des Nutzers.
- Lies `pyproject.toml`, `config.py` (Kommentare enthalten Tuning-Historie!), `main.py`, sowie alle 10 Testdateien in `tests/`, um den Ist-Zustand zu verifizieren — nicht blind dem Audit-Report vertrauen, der Code kann sich seitdem weiterentwickelt haben.

### Phase 1: Re-Audit / Bestandsaufnahme
Erstelle eine aktualisierte, kurze Bestandsaufnahme (kein neues 500-Zeilen-Dokument, sondern eine fokussierte Checkliste) mit dem Status jedes bekannten offenen Punkts aus dem letzten Audit (siehe Tabelle unten) — verifiziert gegen den aktuellen Code, nicht nur übernommen.

Bekannte offene/kritische Punkte aus dem Audit vom 2026-07-02 (Stand meiner Analyse — verifiziere selbst):

| ID | Befund | Status (verifizieren!) |
|---|---|---|
| **S1** | **Kritisch, ungelöst**: Native Access-Violation (`0xC0000005`) in `RoundDriver.select` (`engine/mcts_core.cpp`, aufgerufen aus `mcts/tree.py:1153`). Tritt selbst mit `cpp_round_threads=1` (serialisiert) auf → deutet auf echte Heap-Korruption (z.B. Node-Pointer-Lifetime-Problem mit `py::nodelete` in Kombination mit `advance_root`), nicht auf eine reine Race Condition. Aktuell nur durch Serialisierung (`cpp_round_threads=1`, ~10% Durchsatzverlust) und Supervisor-Auto-Restart umgangen. | Offen — höchste Priorität |
| S9 | C++-RNG (`thread_local mt19937_64` in `mcts_core.cpp`, geseedet aus `random_device`) ist nicht deterministisch seedbar → Reproduzierbarkeits-Anspruch des Projekts stimmt nicht für den C++-Pfad. | Offen |
| S10 | Repo liegt in OneDrive → Sync-Konflikte bei atomaren Checkpoint-Writes (`os.replace()`) auf 278MB-Dateien möglich. | Offen (Ops-Risiko) |
| S13 | GRU-History-Encoder ist ein fp32-"Bug-Magnet" (~12% Latenzkosten), Ablation-Schalter (`use_gru_history=False`) existiert, wurde aber nie tatsächlich als A/B-Experiment gefahren. | Offen — leicht zu testen |
| S14 | Toter Code: `ReplayDataset`, `collate_records`, `dataloader_workers`/`prefetch_factor`/`persistent_workers`/`pin_memory` (`training/replay_buffer.py`), `SelfPlayWorker.run_single_game` (`training/self_play.py`), `aux_steps`. | Vermutlich noch offen |
| S6 | Unklar, ob der In-Training-Arena-Call in `main.py` tatsächlich `batched_bs` und `CONFIG.arena_games` korrekt nutzt (vs. dem separaten `tools/run_arena.py`, wo es verifiziert ist). | Zu verifizieren |
| — | Objektive Elo-Kurve gegen einen fixen Stockfish-Anker wird NICHT automatisch alle ~5k Steps geloggt — nur manuell via `tools/elo_vs_stockfish.py`. Die interne Pool-Elo (`training/pbt.py`) ist selbstreferenziell und kann driften. | Offen |
| — | Selfplay-Durchsatz (~26-300 pos/s auf einer RTX 5070) ist laut Audit 300-1000× langsamer als supervised Distillation (~15-30k pos/s aus Stockfish/Puzzle-Daten). Das Projekt hat bereits begonnen, Distillation zum primären Stärke-Treiber zu machen (`tools/seed_teacher.py`, `tools/stockfish_distill.py`, `--phase=sf_distill`/`puzzles`), aber `teacher_game_ratio=0.0` im Self-Play selbst — die Strategie ist ein Halbschritt. | Zu bewerten/vervollständigen |

Wenn du bei der Verifikation Abweichungen zwischen Audit-Behauptung und tatsächlichem Code findest, dokumentiere das explizit (Audit-Reports können veraltet sein).

### Phase 2: Priorisierten Plan erstellen — BEVOR du Code änderst
Erstelle einen konkreten, nach Impact/Risiko sortierten Umsetzungsplan (nutze Plan-Mode falls verfügbar) mit einzelnen, unabhängig committbaren Schritten. Jeder Schritt braucht: was geändert wird, warum, wie es getestet/gemessen wird (vorher/nachher-Metrik, nicht nur "Tests grün"), und das Rückfallrisiko. Lass dir diesen Plan vom Nutzer bestätigen, bevor du große strukturelle Refactors (insbesondere WebUI) beginnst.

---

## Fokusbereich A: Training-Loop & Self-Play

Ziel: Stabilität, Durchsatz und tatsächliche Spielstärke pro Trainingsstunde erhöhen — nicht nur Code-Hygiene.

1. **S1-Crash angehen.** Das ist der wichtigste offene Punkt im gesamten Projekt. Baue eine ASan-Instrumentierung (`/fsanitize=address` in `setup_ext.py`, oder alternativ ein Linux/WSL-ASan-Build der C++-Extension) und reproduziere den Crash gezielt (z.B. via `audit/scripts/stress_train.py` / `bench_mcts.py`, die bereits als Diagnose-Tooling existieren). Finde die Root Cause in `engine/mcts_core.cpp` (Verdacht: Node-Pointer-Lifetime rund um `advance_root`/`py::nodelete`). Erst wenn die Ursache gefunden und behoben ist, `cpp_round_threads` wieder >1 setzen und den Durchsatzgewinn messen.
2. **GRU-Ablation als echtes Experiment fahren.** Trainiere einen Vergleichslauf mit `use_gru_history=False` gegen die aktuelle Baseline (gleiche Steps, gleiche Daten wo möglich) und miss: Sims/s, Wallclock/Step, finale Pool-/SF-Elo. Dokumentiere das Ergebnis im Audit-Verzeichnis und entscheide datenbasiert, ob der GRU-Encoder bleibt.
3. **Objektive Stärkemessung automatisieren.** Baue eine automatische Elo-Messung gegen einen fixen Stockfish-Anker (wiederverwendbar aus `tools/elo_vs_stockfish.py`) alle ~5000 Trainingssteps, geloggt in TensorBoard neben der internen Pool-Elo. Ohne das lässt sich nicht verlässlich beurteilen, ob eine der anderen Änderungen tatsächlich hilft.
4. **Self-Play- vs. Distillation-Strategie vervollständigen.** Bewerte explizit, ob `teacher_game_ratio` weiterhin 0.0 bleiben soll, ob der Teacher-Buffer periodisch mit frischen Stockfish-Partien/stärkeren Engines nachgeseedet werden sollte, und ob die Trainings-Zeit-Allokation (Self-Play vs. `--phase=sf_distill`/`puzzles`) angesichts des 300-1000×-Durchsatzunterschieds sinnvoll ist. Miss den tatsächlichen Elo-Gewinn pro GPU-Stunde für beide Pfade, bevor du das Verhältnis änderst.
5. **RNG-Determinismus fixen (S9).** C++-RNG-Seed aus Python durchreichbar machen, damit Trainingsläufe bei Bedarf reproduzierbar sind.
6. **Toten Code entfernen (S14).** `ReplayDataset`, `collate_records`, DataLoader-bezogene Config-Felder, `run_single_game`, `aux_steps` — entweder sauber entfernen oder (falls ein echter Performance-Case existiert, z.B. paralleles Prefetching während GPU-Forward) tatsächlich verdrahten. Keine Halbfertig-Scaffolds stehen lassen.
7. **Geteilte "Engine-Harness"-Abstraktion schaffen.** `web_gui.py`, `uci_engine.py`, `run_arena.py`, `elo_vs_stockfish.py` und `main.py` implementieren aktuell jeweils eigenständig Checkpoint-Laden, `match_arch`, `build_model`, `build_inference_engine`. Extrahiere das in ein gemeinsames Modul (z.B. `engine_harness.py`), ohne Verhalten zu verändern — das ist Voraussetzung für konsistente Fixes an allen "Spielen gegen die KI"-Pfaden gleichzeitig (siehe Fokusbereich C).
8. **OneDrive-Risiko dokumentieren/mitigieren (S10).** Prüfe, ob Checkpoints/Replay-Buffer-Dumps in einen Nicht-OneDrive-Pfad ausgelagert werden können (z.B. per Config-Option, Default beibehalten aber Empfehlung dokumentieren), um Sync-Locks bei `os.replace()` zu vermeiden.

**Nicht tun:** Architektur des Netzes (`model/network.py`) nicht ohne triftigen, gemessenen Grund verändern — es gab bereits mehrere schmerzhafte Architektur-Migrationen (alter `fc_aux`-Head, alter 2-Channel-Policy-Bottleneck), die im `Trainer.load_checkpoint` noch als Kompatibilitäts-Code mitgeschleppt werden. Jede weitere Architekturänderung braucht einen sauberen Migrationspfad plus Tests.

---

## Fokusbereich B: Web-UI (`tools/web_gui.py`)

Ist-Zustand: 3291 Zeilen, reines `http.server`/`BaseHTTPRequestHandler` (kein Flask/FastAPI), Frontend als ein einziger großer inline-HTML/CSS/JS-String, ein einziges globales `GameSession`-Objekt (`SESSION`) für genau eine lokale Session — explizit im Docstring als "one server = one session" dokumentiert. Funktional aber bereits sehr feature-reich: SSE-Live-Streaming der Suche, Eval-Bar, Top-Moves mit Visit-Counts/Q, PV, Ponder (überlebt über `tree.advance()`), Kommentare zu menschlichen Zügen, Search-Tree-Visualizer, Takeback (per FEN-Replay), Trainings-Monitor-Panel (liest TensorBoard-Events + `pool_state.json` direkt von Platte).

Ziel: Diese bereits gute UX-Substanz erhalten, aber die technische Basis production-tauglich machen.

1. **Backend/Frontend trennen und strukturieren.** Löse den 3291-Zeilen-Monolithen auf: Backend als sauber geroutete API (Flask/FastAPI oder zumindest ein strukturiertes Routing statt manuellem `if path == ...`-Dispatch), Frontend als eigene Datei(en)/Assets statt inline-String-Literal. Kein Big-Bang-Rewrite — inkrementell migrieren, nach jedem Schritt manuell im Browser verifizieren (Golden Path: neues Spiel starten, Zug machen, KI antwortet, SSE-Stream zeigt Suche, Takeback, Resign).
2. **Mehrere gleichzeitige Sessions/Spiele ermöglichen** (zumindest mehrere Browser-Tabs/Games ohne dass sie sich gegenseitig Board-State überschreiben) — aktuell durch das globale `SESSION`-Objekt unmöglich. Session-Keying (z.B. per Cookie/Session-ID) statt eines einzigen globalen Zustands.
3. **Engine-/Checkpoint-Auswahl in der UI.** Aktuell ist die KI-Stärke nur über Sims/Zeit/DCA regelbar, es gibt keine Möglichkeit, einen anderen Checkpoint oder ein bestimmtes Opponent-Pool-Mitglied auszuwählen. Baue eine UI-Auswahl (Dropdown), die über die Engine-Harness aus Fokusbereich A verschiedene Checkpoints laden kann, inkl. sichtbarer Stärkeindikation (z.B. bekannte SF-Elo aus der automatisierten Messung in Fokusbereich A.3).
4. **Persistenz.** Spielverlauf/Settings überleben aktuell keinen Server-Neustart (nur In-Memory). Mindestens Spielverlauf/PGN-Export persistieren.
5. **Sicherheits-/Deployment-Realität berücksichtigen.** Es existieren `start_webgui_ngrok.bat`/`ngrok_url.txt` im Repo — die UI wird also mindestens gelegentlich öffentlich exponiert, obwohl sie keinerlei Auth hat. Kläre mit dem Nutzer, ob externer Zugriff ein echter Use-Case ist; falls ja, minimal Auth (z.B. simples Token/Passwort) ergänzen, bevor größere Refactors darauf aufbauen.
6. **Threading-Modell überprüfen.** Der Single-`RLock`-pro-Session-Ansatz (Ponder-Thread + Such-Thread + SSE-Broadcaster) ist fragil bei Erweiterung — beim Multi-Session-Umbau (Punkt 2) muss das neu durchdacht werden, nicht einfach vervielfacht.
7. **Frontend-Qualität.** Kein Build-Tool nötig, aber zumindest grundlegende Struktur (separate JS-Datei(en), kein tausende-Zeilen-String), und wo sinnvoll leichte Tests/Smoke-Checks für die kritischen API-Routen (`/api/move`, `/api/new`, `/api/state`).

**Nicht tun:** Die bereits gute UX (Ponder, Kommentar-Heuristiken, Tree-Visualizer, Live-Trainings-Monitor) nicht wegrefactorn oder vereinfachen, nur weil der Code umstrukturiert wird — Funktionalität muss vor/nach dem Refactor 1:1 im Browser verglichen werden.

---

## Fokusbereich C: Spielen gegen die KI (`main.py --play`, `tools/uci_engine.py`, WebUI-Spielpfad)

Ist-Zustand: Drei unabhängige Implementierungen mit unterschiedlicher Qualität.

1. **`uci_engine.py` auf den gebatchten Suchpfad umstellen.** Aktuell läuft die Suche dort Leaf-für-Leaf (`run_one_simulation` in einer Schleife) statt über den gebatchten/CUDA-Graph-Pfad, den WebUI und Self-Play nutzen — das macht UCI-GUI-Partien (Arena, Cute Chess, Nibbler) unnötig langsam/schwach im Vergleich zur eigentlichen Modellstärke. Nutze die Engine-Harness aus Fokusbereich A.7, um Konsistenz herzustellen.
2. **UCI-Protokoll vervollständigen:** echtes `ponderhit`-Handling (aktuell nur `infinite`-artig, Ponder-Züge werden nicht wirklich genutzt), `MultiPV`, und — wichtig für Nutzbarkeit gegen menschliche Spieler unterschiedlicher Stärke — eine Form von Stärkebegrenzung analog zu `UCI_LimitStrength`/`UCI_Elo` (kann auf Basis der in Fokusbereich A.3 etablierten SF-Elo-Kalibrierung umgesetzt werden, z.B. durch Sim-Anzahl/Noise-Mapping auf Ziel-Elo).
3. **Konsistente Stärke-/Zeitkontrolle über alle drei Pfade** (Terminal, UCI, WebUI) — aktuell hat jeder Pfad eigene, leicht unterschiedliche Logik für Sims/Zeit/Ponder. Nach der Harness-Extraktion (A.7) sollte Zeitkontroll-Logik (inkl. clock-basiertem Budget wie in `uci_engine.py` bereits vorhanden: `clock/25 + inc*0.75`) an einer Stelle liegen und von allen drei Frontends genutzt werden.
4. **End-to-End-Verifikation:** Nach den Änderungen tatsächlich eine Partie in einer UCI-GUI (z.B. Arena/Cute Chess) UND im Web-UI spielen und beobachten — nicht nur Unit-Tests. Insbesondere Ponder, Takeback, Resign, und Zeitmanagement unter echten Bedingungen prüfen.

---

## Arbeitsweise & Leitplanken

- **Kleine, unabhängig testbare Commits.** Kein Mega-Commit über alle drei Fokusbereiche. Jeder Commit: eine zusammenhängende Änderung, mit Vorher/Nachher-Messung wo relevant (Sims/s, Latenz, Testergebnisse), referenziert die Audit-ID falls zutreffend (Projektkonvention: `# audit S1` etc. als Kommentar am Fix-Ort).
- **Tests zuerst, dann Fix, dann verifizieren.** Nutze die bestehende Testsuite (`pytest tests/`) und die Diagnose-Skripte in `audit/scripts/` (z.B. `stress_train.py`, `bench_mcts.py`, `profile_selfplay.py`) — sie existieren bereits und sollten wiederverwendet, nicht neu erfunden werden. Ergänze `tests/test_audit_regressions.py` um Regressionstests für jeden neuen Fix.
- **Nichts destruktiv löschen ohne Rückfrage:** Checkpoints (`checkpoints/`), Replay-Buffer, `runs/`-Logs, `latest.pt.collapsed` (bewusst aufbewahrtes Artefakt eines früheren Trainingskollapses) nicht löschen. Kein `git push --force`, kein `--no-verify`.
- **CI beachten:** `.github/workflows/ci.yml` baut die C++-Extension und läuft pytest auf Ubuntu+Windows/py3.11 CPU-only. Der `lint`-Job (ruff) ist aktuell `continue-on-error: true` — wenn du das im Rahmen der Aufräumarbeiten scharfschaltest, tu das erst nachdem der Code lint-sauber ist, nicht vorher.
- **Windows/cp1252-Falle:** Neue Tools/Skripte müssen `sys.stdout`/`sys.stderr` explizit auf UTF-8 reconfiguren, sonst crasht Output mit Nicht-ASCII-Zeichen unter `.bat`/Redirect (bereits mehrfach im Projekt passiert, siehe `8f2c599`).
- **Dokumentiere Ergebnisse dort, wo das Projekt es erwartet:** neue Findings/Fixes als kurzer Eintrag im `audit/`-Verzeichnis (Konvention: `AUDIT_REPORT_<Datum>.md` oder `findings/`), nicht als Ad-hoc-Analyse-Dateien im Hauptverzeichnis.
- **Melde dich mit einem konkreten Plan zurück, bevor du den großen WebUI-Refactor (Fokusbereich B.1/B.2) beginnst** — das ist der risikoreichste, am schwersten reversible Teil dieser Aufgabe.

---

## Abnahmekriterien

Am Ende sollte belegbar sein (nicht nur behauptet):
- S1-Crash entweder behoben (mit ASan-Beweis) oder mit klar dokumentierter, tiefergehender Root-Cause-Analyse, falls nicht vollständig lösbar in der verfügbaren Zeit.
- Eine automatisierte, in TensorBoard sichtbare objektive Elo-Kurve gegen Stockfish.
- Ergebnis des GRU-Ablation-Experiments, mit Zahlen, nicht nur Meinung.
- Web-UI: bestehende Features funktionieren nachweislich weiter (manuell im Browser getestet), plus mindestens Multi-Session-Fähigkeit und Checkpoint-Auswahl.
- Alle drei "Spielen gegen die KI"-Pfade nutzen denselben gebatchten Suchkern und dieselbe Zeitkontroll-Logik.
- `pytest tests/` weiterhin grün, neue Regressionstests für jeden behobenen Bug.
