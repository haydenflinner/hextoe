# Status
This is a Rust hexo engine, it's pretty fast. It has Python bindings, if you'd like to code your bot in Python.
It also has a Chrome console-paste hook which forwards online play to the bot for live feedback.

It doesn't seem to be better than smart people, yet, but the browser-to-bot sync is nice and it _should_ be feasible as a base for someone who knows what they're doing.

One obvious improvement is to consider things as one turn with two plays within the turn, rather than
as two independent turns that have to be considered separately (which really wastes compute considering things as different when executed in different order when they're the same).


the last day of toil seemed to only make things worse, I think this git sha was probably when it last performed pretty well

○  utkttmzx hayden 2026-03-28 08:59:59 972601f0 <-- GIT SHA
│  oh hell yeah


# Features
  1. bookmarklet.js -- scrapes games and downloads them
  2. interceptor.js -- watches your current game and forwards moves over to the bot
  3. various training binaries
  4. Main bot:
    `cargo run --release --bin hextoe`

# Rust portion:
just use rustup and you're good to go

# Python portion:
source .venv/bin/activate
uv pip install maturin safetensors torch numpy
uv pip install safetensors
uv pip install packaging

    36s    1d ago python train.py games-1774572188158.json
    39s    1d ago python train.py game-chunks/ih3t-games-chunk*
  9 5m     1d ago python train.py game-chunks/ih3t-games-chunk-*1*.json
  8 47s    1d ago python train_selfplay.py --naive-frac 0.95

