# Lichess Bot — Setup Guide

> **Status (geprüft 2026-06-15):** `tools/lichess_bot.py` ist funktionsfähig und nutzt
> aktuelle, offizielle Lichess-Bot-API-Endpunkte. Keine externen Abhängigkeiten nötig
> — reines Python-stdlib (`urllib`). Bekannte Einschränkungen sind weiter unten dokumentiert.

---

## Inhaltsverzeichnis

1. [Voraussetzungen](#1-voraussetzungen)
2. [Lichess BOT-Account erstellen](#2-lichess-bot-account-erstellen)
3. [API-Token generieren](#3-api-token-generieren)
4. [Token im Projekt konfigurieren](#4-token-im-projekt-konfigurieren)
5. [Bot starten](#5-bot-starten)
6. [Checkpoint auswählen](#6-checkpoint-auswählen)
7. [Vorbereitende Fixes (empfohlen vor erstem echten Einsatz)](#7-vorbereitende-fixes-empfohlen-vor-erstem-echten-einsatz)
8. [Troubleshooting](#8-troubleshooting)

---

## 1. Voraussetzungen

- Python-Umgebung mit allen Abhängigkeiten aus `requirements.txt` ist aktiv
- Mindestens ein trainierter Checkpoint existiert in `checkpoints/step_*.pt`
- Kein laufendes Training auf demselben Checkpoint (lesen-gleichzeitig ist OK; das Skript
  öffnet den Checkpoint read-only)

---

## 2. Lichess BOT-Account erstellen

Lichess-BOT-Accounts sind **separate** Accounts — du kannst deinen persönlichen Spieler-Account
**nicht** zu einem BOT upgraden (sobald er Partien gespielt hat). Gehe so vor:

### 2a. Neuen Account registrieren

1. Gehe zu **https://lichess.org** und klicke **Sign in → Register**
2. Wähle einen Bot-Namen, z. B. `ChatKi-Bot` oder `YourNameAI`
3. E-Mail bestätigen

> ⚠ Der Account darf **null gespielte Partien** haben, wenn du ihn upgradest.
> Ein frisch registrierter Account erfüllt das automatisch.

### 2b. Account zu BOT upgraden

Das Upgrade erfolgt per API-Aufruf (einmalig, **nicht rückgängig zu machen**).
Führe das nach dem Erstellen des API-Tokens aus (Schritt 3):

```powershell
# Windows PowerShell — ersetze <TOKEN> durch dein echtes Token
curl.exe -X POST https://lichess.org/api/bot/account/upgrade `
  -H "Authorization: Bearer <TOKEN>"
```

Erfolgreiche Antwort:

```json
{"ok": true}
```

---

## 3. API-Token generieren

1. Melde dich auf lichess.org **mit dem BOT-Account** an
2. Gehe zu **https://lichess.org/account/oauth/token/create**
3. Beschreibung: z. B. `ChatKi Bot Token`
4. Aktiviere den Scope **`bot:play`** (alle anderen können deaktiviert bleiben)
5. Klicke **Submit** — das Token wird **einmalig** angezeigt, sofort kopieren!

> ⚠ Das Token hat vollen Zugriff auf den Bot-Account. Teile es nicht,
> committe es nicht in Git. Speichere es sicher (z. B. im Windows Credential Manager
> oder in einer `.env`-Datei, die in `.gitignore` steht).

---

## 4. Token im Projekt konfigurieren

Das Skript liest das Token aus der **Umgebungsvariable `LICHESS_TOKEN`**
oder dem CLI-Flag `--token`. Empfohlen: Umgebungsvariable, so bleibt der Token
aus der Shell-History und dem Code heraus.

### Option A — Für diese PowerShell-Session (temporär)

```powershell
$env:LICHESS_TOKEN = "lip_DEIN_TOKEN"
```

### Option B — Dauerhaft in Windows (für den aktuellen Benutzer)

```powershell
[System.Environment]::SetEnvironmentVariable("LICHESS_TOKEN", "lip_DEIN_TOKEN", "User")
# Danach neue PowerShell-Session öffnen
```

### Option C — `.env`-Datei (lokal, nicht committen)

Erstelle `chess_ai/.env`:

```
LICHESS_TOKEN=lip_DEIN_TOKEN
```

Dann vor dem Start:

```powershell
Get-Content .env | ForEach-Object { if ($_ -match '^(\w+)=(.*)$') { [System.Environment]::SetEnvironmentVariable($matches[1], $matches[2]) } }
```

Sicherstellen, dass `.env` in `.gitignore` steht:

```
.env
```

---

## 5. Bot starten

Aus dem Projektverzeichnis `chess_ai/`:

```powershell
# Einfachster Start — neuesten Checkpoint und Standard-Sims (128) nutzen
python tools/lichess_bot.py

# Mit explizitem Checkpoint und erhöhten Sims für stärkeres Spiel
python tools/lichess_bot.py --resume checkpoints/step_0021200.pt --sims 200

# Token direkt übergeben (weniger empfohlen — erscheint in Shell-History)
python tools/lichess_bot.py --token lip_DEIN_TOKEN --sims 200
```

Erwartete Ausgabe beim Start:

```
[ckpt] Checkpoint has 10 residual blocks (config 10) — matching architecture.
[lichess] Loaded checkpoints/step_0021200.pt
[lichess] Logged in as yourbot-id. Waiting for challenges…
```

Danach wartet der Bot auf eingehende Challenges. Schicke eine Challenge über die Lichess-UI
oder von einem anderen Account. Der Bot akzeptiert automatisch **ungewertete Standard-Partien**
und lehnt alles andere ab.

---

## 6. Checkpoint auswählen

Das Skript verwendet automatisch den neuesten `step_*.pt`-Checkpoint, wenn `--resume`
weggelassen wird (via `utils/ckpt_arch.latest_checkpoint()`).

Aktuell neuester Checkpoint: **`checkpoints/step_0021200.pt`** (Stand: 2026-06-15,
Training-Schritt ~21.200 im aktuellen Fresh-Run nach dem Canonical-Encoding-Reset vom 2026-06-12).

| Checkpoint | Training-Stand | Geschätzte Stärke |
|---|---|---|
| `step_0005000.pt` | Frühes Training | sehr schwach |
| `step_0010000.pt` | ~1 h Training | schwach |
| `step_0021200.pt` | Aktuell neuester | schwach bis mittel |

Die Architektur wird beim Laden automatisch an den Checkpoint angepasst (`match_arch`);
es gibt kein manuelles Eingreifen nötig, auch wenn sich Hyperparameter geändert haben.

---

## 7. Vorbereitende Fixes (empfohlen vor erstem echten Einsatz)

Der Code ist grundsätzlich funktionsfähig, hat aber bekannte Einschränkungen.
**Hier wird nur beschrieben, was geändert werden müsste — keine Implementierung.**

### Fix 1 — Zeitkontrolle fehlt komplett (kritisch!)

**Problem:** Das Skript schaut die Uhr nicht an. Bei schnellen Zeitkontrollen
(Bullet / Blitz) kann der Bot auf Zeit verlieren, weil `tree.search()` mit festen
`--sims` läuft, egal wie viel Restzeit noch da ist.

**Lösung:** Im `state`-Dict stehen `wtime`, `btime`, `winc`, `binc` (Millisekunden).
Bevor ein Zug berechnet wird, sollte `n_sims` dynamisch reduziert werden, wenn die
Restzeit gering ist. Grobe Heuristik: `sims = max(50, base_sims * (remaining_ms / 30000))`.

> **Empfehlung für erste Tests:** Nur **Correspondence** (unbegrenzte Zeit) oder
> **Classical** (≥15 min) Challenges annehmen, bis die Zeitkontrolle implementiert ist.
> Setze die Accept-Bedingung in Zeile 169 auf `ch["speed"] == "correspondence"` oder filtere
> nach `ch["timeControl"]["limit"] >= 900` (≥15 min).

### Fix 2 — Keine Reconnect-Logik (robustness)

**Problem:** Wenn der Event-Stream `/api/stream/event` abbricht (Netzwerktimeout,
Lichess-Neustart), stirbt der gesamte Bot-Prozess.

**Lösung:** Den `for ev in li.stream(...)` in der `main()`-Funktion in eine
`while True`-Schleife mit `try/except` und `time.sleep(5)` wrappen.

### Fix 3 — Rated-Game-Filter (optional, nach Geschmack)

**Problem:** Alle gewerteten Challenges werden abgelehnt (`not ch.get("rated", False)`).

**Lösung:** Bedingung entfernen oder aufweichen, wenn du gewertete Partien spielen möchtest.
Beachte: Gewertete Partien beeinflussen die Rating-Geschichte des BOT-Accounts dauerhaft.

### Fix 4 — gameFull mit nicht-gestarteten Partien (edge case)

**Problem:** Wenn ein `gameFull`-Event ankommt, dessen `state.status` nicht `"started"` ist
(z. B. Aborted), wird das im aktuellen Code still ignoriert, aber die Schleife läuft weiter.
Kein echter Bug, aber schmutziges Verhalten.

**Lösung:** Statusprüfung auch beim ersten `gameFull`-Event vor dem Setzen von `my_white`.

---

## 8. Troubleshooting

### "No token. Set LICHESS_TOKEN or pass --token"

→ Token nicht gesetzt. Siehe [Schritt 4](#4-token-im-projekt-konfigurieren).

### `urllib.error.HTTPError: HTTP Error 401: Unauthorized`

→ Token ungültig oder abgelaufen. Neues Token auf lichess.org erstellen.
Token-Scope `bot:play` prüfen.

### `urllib.error.HTTPError: HTTP Error 403: Forbidden` beim Upgrade

→ Account hat bereits gespielte Partien und kann nicht mehr zu BOT werden.
Neuen Account erstellen.

### `[lichess] No checkpoint — UNTRAINED net.`

→ Kein `step_*.pt` in `checkpoints/` gefunden. Training läuft noch? Expliziten
Pfad via `--resume checkpoints/step_000XXXX.pt` angeben.

### `KeyError: 'model'` beim Checkpoint-Laden

→ Alter Pre-Encoding-Checkpoint (vor 2026-06-12-Reset). Diese sind archiviert
und inkompatibel mit der aktuellen Architektur. Nur `step_*.pt`-Dateien aus dem
aktuellen `checkpoints/`-Ordner verwenden.

### Bot läuft aber nimmt keine Challenges an

1. Stelle sicher, dass der Challenge-Sender den Bot **direkt challengt**
   (Lichess → Profil des Bots → Challenge)
2. Prüfe die Logs: Wird eine Challenge-Zeile ausgegeben? Steht darin `declined`?
   Wenn `"rated": True` oder `"variant" != "standard"`, wird abgelehnt.
3. Auf Lichess müssen BOT-Accounts explizit challengbar sein — prüfe die
   Bot-Account-Einstellungen unter **https://lichess.org/account/preferences/privacy**.

### `RuntimeError: CUDA out of memory`

→ `--sims` reduzieren (z. B. `--sims 50`) oder Bot auf CPU laufen lassen
(setzt `device=cpu` in `config.py` oder via Systemprobe). Während das Training
läuft, teilt der Bot die GPU — das Training hat Priorität.

### Bot verliert auf Zeit

→ Zeitkontrolle nicht implementiert. Siehe [Fix 1](#fix-1--zeitkontrolle-fehlt-komplett-kritisch).
Bis dahin nur Correspondence-Partien spielen.

---

## Kurzreferenz — Alle Flags

```
python tools/lichess_bot.py [--token TOKEN] [--resume CHECKPOINT] [--sims N]

  --token    Lichess API-Token (default: $env:LICHESS_TOKEN)
  --resume   Pfad zum Checkpoint (default: neuester step_*.pt in checkpoints/)
  --sims     MCTS-Simulationen pro Zug (default: CONFIG.mcts_sims = 128)
```
