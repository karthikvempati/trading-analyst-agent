# Trading Analyst Agent — self-hosted

A single-service web app: type a ticker, get a structured trading read
(market regime gate → technicals → valuation → catalysts → entry or
cash-secured-put evaluation → risk-manager veto → GO / CAUTION / NO-GO verdict).

- **Backend:** Python 3.11+ / FastAPI, one file (`app.py`)
- **Frontend:** one static page (`static/index.html`), no build step, phone-friendly
- **Data:** yfinance only — no API keys anywhere. Missing data renders as
  "unavailable", never invented.
- **State:** none. No database, no login by default (it runs behind your server).

> Every conclusion sentence is composed from that run's actual numbers
> (ticker, price vs EMA stack, RSI, volume, regime votes, the specific checks
> that fired). Stock, entry, and CSP modes each produce genuinely different
> verdicts. The CSP math uses only the terms you type — no options chain is
> ever invented.

---

## 1. Run it locally (2 minutes)

```bash
cd trading-analyst-agent-selfhost
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/uvicorn app:app --host 127.0.0.1 --port 8000
```

Open http://127.0.0.1:8000 — or hit the API directly:

```bash
# analyze a stock
curl "http://127.0.0.1:8000/api/analyze?ticker=ADBE&mode=stock"

# evaluate an entry
curl "http://127.0.0.1:8000/api/analyze?ticker=ADBE&mode=entry"

# evaluate a cash-secured put (terms come from YOUR broker chain)
curl "http://127.0.0.1:8000/api/analyze?ticker=ADBE&mode=csp&strike=230&credit=4.20&delta=0.25&iv=0.52&dte=34&contracts=2"
```

Smoke test (boots its own server, checks 16 assertions incl. the
"same-response" regression):

```bash
.venv/bin/python test_smoke.py
```

## 2. Deploy on a fresh Ubuntu/Debian server with Docker + Caddy (recommended)

You need: a server with a public IP, and a domain whose DNS **A record**
points at that IP (e.g. `trading.example.com` → your server IP). The
commands below assume Ubuntu 22.04/24.04 and a domain you own.

**On the server, as a user with sudo:**

```bash
# 1) Install Docker (official convenience script)
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER
# log out and back in so the docker group applies, then:
docker --version

# 2) Install Caddy (serves HTTPS automatically)
sudo apt install -y debian-keyring debian-archive-keyring apt-transport-https curl
curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/gpg.key' | sudo gpg --dearmor -o /usr/share/keyrings/caddy-stable-archive-keyring.gpg
curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/debian.deb.txt' | sudo tee /etc/apt/sources.list.d/caddy-stable.list
sudo apt update
sudo apt install -y caddy

# 3) Copy this project onto the server (run from your laptop)
scp -r trading-analyst-agent-selfhost your-user@YOUR_SERVER_IP:/home/your-user/
# then on the server:
cd ~/trading-analyst-agent-selfhost

# 4) Point Caddy at your domain (edit the example, save as the live file)
sed 's/trading.example.com/trading.YOURDOMAIN.com/' deploy/Caddyfile.example | sudo tee /etc/caddy/Caddyfile
# ^ replace trading.YOURDOMAIN.com with your real domain in that command
sudo systemctl reload caddy

# 5) Start the app
docker compose up -d --build
docker compose ps
```

Open `https://trading.YOURDOMAIN.com` — Caddy fetches the certificate
automatically on first visit. Check logs any time with
`docker compose logs -f`.

**Updating later:** copy the new files over, then
`docker compose up -d --build`.

### Optional: require a login (basic auth)

```bash
caddy hash-password   # paste the hash into /etc/caddy/Caddyfile (see the commented basicauth block)
sudo systemctl reload caddy
```

## 3. Alternative: nginx + Certbot (instead of Caddy)

```bash
sudo apt install -y nginx certbot python3-certbot-nginx
cd ~/trading-analyst-agent-selfhost
docker compose up -d --build          # app on 127.0.0.1:8000
# edit the domain in the example, then:
sudo sed 's/trading.example.com/trading.YOURDOMAIN.com/' deploy/nginx-example.conf \
  | sudo tee /etc/nginx/sites-available/trading-analyst
sudo ln -s /etc/nginx/sites-available/trading-analyst /etc/nginx/sites-enabled/
sudo nginx -t && sudo systemctl reload nginx
sudo certbot --nginx -d trading.YOURDOMAIN.com
```

## 4. Alternative: systemd without Docker

```bash
sudo mkdir -p /opt/trading-analyst
sudo cp -r . /opt/trading-analyst
cd /opt/trading-analyst
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
# edit User= and WorkingDirectory= in deploy/trading-analyst.service to match your setup
sudo cp deploy/trading-analyst.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now trading-analyst
systemctl status trading-analyst
```

Then put Caddy or nginx in front as in sections 2/3.

## 5. Notes

- **Behind an egress proxy?** Set `YF_PROXY=http://user:pass@proxy:port`
  (or just export the standard `HTTPS_PROXY`) before starting — the app
  honors it for Yahoo fetches.
- **Ports:** the app listens on 8000 inside Docker; the compose file maps
  host 8000 → container 8000. Caddy/nginx terminate TLS and proxy to it.
- **Performance:** one analysis makes ~8 Yahoo fetches in parallel and
  typically completes in 5–15 seconds on a normal connection.
- **No auth by default** — bind it to 127.0.0.1 and proxy through
  Caddy/nginx (as above), and/or enable the basic-auth snippets.
- **Not financial advice.** The verdicts are deterministic reads of public
  quote data; verify CSP terms and earnings dates on your broker/company IR
  before trading.

## API reference

`GET /api/analyze?ticker=ADBE&mode=stock|entry|csp`

| param     | required | notes                                                        |
|-----------|----------|--------------------------------------------------------------|
| ticker    | yes      | e.g. `ADBE`                                                  |
| mode      | no       | `stock` (default), `entry`, or `csp`                          |
| strike    | csp only | strike price, e.g. `230`                                     |
| credit    | csp only | premium received per share, e.g. `4.20`                      |
| delta     | no       | e.g. `0.25` — screened against ≤ 0.30                        |
| iv        | no       | e.g. `0.52` for 52%                                          |
| dte       | no       | days to expiry (default 30)                                  |
| contracts | no       | default 1                                                    |

Response JSON: `verdict` (GO/CAUTION/NO-GO), `verdict_headline`,
`regime`, `technicals`, `valuation`, `catalysts`, `mode_evaluation`
(+ `metrics` for CSP), `risk` (blockers/warnings), all with `as_of`
timestamps. `GET /api/health` → `{"ok": true}`.

The web UI also includes an interactive daily candlestick chart with EMA20,
EMA50, volume, RSI, the Chaikin Accumulation/Distribution Line, and 20-day
Chaikin Money Flow. Select `3mo`, `6mo`, `9mo`, `1y`, or `2y`; use the chart
controls or mouse wheel to zoom and drag to pan. Chart data is available at
`GET /api/chart?ticker=ADBE&period=9mo`.
