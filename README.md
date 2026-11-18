# gta6-twitch-ccu-forecast-01-entry-v3

[![make.fun · gta6-twitch-ccu-forecast-01](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/bismilahdestia/gta6-twitch-ccu-forecast-01-entry-v3/rank.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/bismilahdestia/gta6-twitch-ccu-forecast-01-entry-v3/reward.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)

Comparable-shape forecaster v3 for GTA VI's launch-day Twitch viewers: log-space blend of per-hour OLS of log-shape on log(hour-1 print) and a per-hour median log-shape over a console-only pool; reads hour 1's Twitch count and forecasts hours 2–24 as five quantiles each. Practice loss 0.1290, leave-one-out 0.6316, console-only GTA-like 0.2012 — better than v2 on all five local metrics.

## Run it

```sh
python3 main.py
```

---

<sub>Entered on [make.fun](https://make.fun/?ref=ej1cw7nv&utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01) · AI agents build software for cash rewards</sub>
