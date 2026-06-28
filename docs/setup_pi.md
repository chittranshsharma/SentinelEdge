# Raspberry Pi 4 — SentinelEdge Setup Guide

## Prerequisites
- Raspberry Pi 4 (any RAM)
- Raspberry Pi OS Lite (64-bit, Bookworm)
- Network access (Ethernet preferred for stability)

---

## 1. System Update

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y python3.11 python3.11-venv python3-pip git curl
```

---

## 2. Mosquitto MQTT Broker

```bash
sudo apt install -y mosquitto mosquitto-clients
```

Edit the config:
```bash
sudo nano /etc/mosquitto/conf.d/sentineledge.conf
```

Add:
```
# Standard MQTT listener
listener 1883
allow_anonymous true

# WebSocket listener (for browser dashboard debugging)
listener 9001
protocol websockets
allow_anonymous true
```

Enable and start:
```bash
sudo systemctl enable mosquitto
sudo systemctl restart mosquitto
sudo systemctl status mosquitto

# Test it works:
mosquitto_sub -t "test/#" -v &
mosquitto_pub -t "test/hello" -m "world"
```

---

## 3. Clone Repository

```bash
cd ~
git clone https://github.com/yourname/sentineledge.git
cd sentineledge/backend
```

---

## 4. Python Virtual Environment

```bash
python3.11 -m venv ~/sentineledge/venv
source ~/sentineledge/venv/bin/activate
pip install -r requirements.txt
```

---

## 5. Environment Variables

```bash
cp .env.example .env
nano .env
```

Fill in:
```env
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_SERVICE_KEY=your-service-role-key
GROQ_API_KEY=gsk_...
TELEGRAM_BOT_TOKEN=...
TELEGRAM_CHAT_ID=...
MQTT_BROKER_HOST=localhost
MQTT_BROKER_PORT=1883
DEVICE_ID=sentineledge-001
```

---

## 6. Supabase Schema

Copy `supabase_schema.sql` contents into the Supabase SQL Editor and run.

Then enable Realtime:
- Supabase Dashboard → Database → Replication
- Add `sensor_readings` and `anomalies` to realtime publication

---

## 7. Test Each Module

```bash
source ~/sentineledge/venv/bin/activate
cd ~/sentineledge/backend

# Test Supabase connection
python -c "from supabase_client import get_client; c=get_client(); print('Supabase OK')"

# Test Groq
python -c "import asyncio; from groq_client import generate_explanation; asyncio.run(generate_explanation('imbalance', 0.93, 2.34, 34.2, 61, 320, 28.6, 77.2, 1700000000))"

# Test Telegram
python -c "import asyncio; from telegram_client import send_telegram_alert; asyncio.run(send_telegram_alert('🔧 SentinelEdge backend test message'))"
```

---

## 8. Run Backend

```bash
# Test run (Ctrl+C to stop)
python main.py

# You should see:
# INFO:     Supabase connection verified
# [MQTT] Connected to localhost:1883
# INFO:     Application startup complete.
```

---

## 9. Systemd Auto-Start

```bash
sudo cp ~/sentineledge/backend/systemd/sentinel-backend.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable sentinel-backend
sudo systemctl start sentinel-backend

# Check logs
journalctl -u sentinel-backend -f
```

---

## 10. Find Pi's IP Address (for ESP32 config)

```bash
hostname -I
```

Use this IP in `firmware/src/main.cpp` as `MQTT_HOST`.

---

## Troubleshooting

| Issue | Check |
|-------|-------|
| MQTT not receiving messages | `mosquitto_sub -t "sentineledge/#" -v` and watch serial from ESP32 |
| Supabase writes fail | Verify service key (not anon key) and table exists |
| Groq timeout | Check GROQ_API_KEY; rate limits on free tier |
| Telegram not sending | Verify bot token and chat_id (send `/start` to bot first) |
