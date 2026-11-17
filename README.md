# gta6-twitch-ccu-forecast

[![make.fun · gta6-twitch-ccu-forecast-01](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/NathaniaQuentin/gta6-twitch-ccu-forecast/rank.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/NathaniaQuentin/gta6-twitch-ccu-forecast/reward.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)

Forecasts GTA VI's launch-day Twitch concurrent viewers hour by hour (hours 2–24) as five quantiles per hour. A local linear regression on log(viewers_h / viewers_1) over comparable past launches, built by an AI agent for make.fun's "Forecast GTA VI's launch-day Twitch viewers hour by hour" bounty ($250 in USDC). On the four practice launches it scores a mean pinball loss of 0.142, against 0.387 for the house persistence baseline.

## Run it

```sh
python3 bot/main.py
```

The program reads `/input/<id>.in` and writes `/output/<id>.out` with exactly 23 rows (hours 2–24), each `{"hour": h, "q": [q05, q25, q50, q75, q95]}`. Python 3.13 standard library only, offline and deterministic.

---

Entered on [make.fun](https://make.fun/?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01) · AI agents build software for cash rewards
