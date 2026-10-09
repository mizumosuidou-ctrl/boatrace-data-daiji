#!/usr/bin/env bash
# 学習系のコマンドを「裏で・1つだけ」実行する（画面を閉じても続く。動いていれば何もしない。cron の学習とも重ならない）。
#   使い方（ubuntu で）:  bash /opt/minamo/deploy/ml_bg.sh ml-train     （ml-tune なども同じ）
#   経過:  tail -5 ~/ml_bg_ml-train.log     （ログは ubuntu のホームに書く。/opt/minamo/var は root のものなので書けない）
set -uo pipefail
cd "$(dirname "$0")"
cmd="${1:-}"
case "$cmd" in
  ml-train|ml-tune|ml-years|ml-softmax|ml-softmax-roi|ml-calib) ;;
  *) echo "使い方: ml_bg.sh ml-train|ml-tune|ml-years|ml-softmax|ml-softmax-roi|ml-calib"; exit 2 ;;
esac
if pgrep -f "minamo ml-" >/dev/null; then echo "すでに学習系の処理が動いています。何もしません"; pgrep -af "minamo ml-" | cut -c1-90; exit 1; fi
log="$HOME/ml_bg_${cmd}.log"
nohup bash -c 'echo "開始 $(date +%T)"; sudo flock -n /tmp/minamo-ml.lock docker compose run --rm worker python -m minamo "$@"; rc=$?; echo "終了 $(date +%T) 終了コード $rc"' _ "$@" > "$log" 2>&1 &
sleep 4
if pgrep -f "minamo ml-" >/dev/null; then
  echo "裏で開始しました（ログ: $log）"; head -1 "$log"
else
  echo "開始できませんでした。ログ:"; cat "$log" 2>&1 | head -5; exit 1
fi
