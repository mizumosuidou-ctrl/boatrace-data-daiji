#!/usr/bin/env bash
# 最新の main に更新して、worker を作り直す（学習の最中でも安全：学習は別のコンテナで動く）。
#   使い方（ubuntu で。sudo は付けない）:  bash /opt/minamo/deploy/update.sh
#   cron の登録（install_cron.sh）は、変わったときだけ別に実行する
set -euo pipefail
cd "$(dirname "$0")/.."
echo "--- 今: $(git branch --show-current) $(git log --oneline | head -1)"
if [ -n "$(git status --porcelain --untracked-files=no)" ]; then
  echo "サーバーのファイルが書き換えられているので、更新しません（git status を見せてください）"; git status --short | head -8; exit 1
fi
git fetch -q origin main
git checkout -q main
git pull -q --ff-only origin main
echo "--- 更新後: $(git branch --show-current) $(git log --oneline | head -1)"
cd deploy
sudo docker compose build -q worker
sudo docker compose up -d worker 2>&1 | tail -2
sleep 10
echo "--- コンテナ"; sudo docker ps --format '{{.Names}} | {{.Status}}'
echo "--- 学習中か"; pgrep -af "minamo ml-" | cut -c1-90 || echo "動いていない"
