#!/usr/bin/env bash
# 朝の確認用（読むだけ。何も変更しない）。サーバーの状態・夜の学習・収集・サイトをまとめて表示する。
#   使い方:  bash /opt/minamo/deploy/morning_check.sh
cd "$(dirname "$0")"
echo "=== 時刻・負荷"; date; uptime; free -h | head -2; df -h / | tail -1
echo "=== コンテナ"; sudo docker ps --format '{{.Names}} | {{.Status}}'
echo "=== 手動で動かした学習（~/train_log3.txt）"
if [ -f ~/train_log3.txt ]; then grep -E "^開始|^終了" ~/train_log3.txt; grep -E "INFO minamo.ml.train" ~/train_log3.txt | cut -c1-150 | tail -6; else echo "（ログなし）"; fi
echo "=== 自動の学習（3:10 の cron）"
for f in /opt/minamo/var/cron_daily.log /opt/minamo/var/cron_weekly.log; do [ -f "$f" ] && { echo "[$f]"; grep -E "===|休みます|失敗|Error|Traceback" "$f" | tail -4; }; done
echo "=== 学習の結果（要約の主な行）"
sudo docker compose run --rm -T worker python -c "import json;from minamo.ml import train;from minamo.ml.live import ML_DIR;m=json.load(open(ML_DIR/'meta.json'));print('学習した時刻:', m.get('trained_at'));print(train.summary_ja(m))" 2>&1 | grep -E "学習した時刻|^LightGBM展示前|^＋今節|^＋ファン|ファン手帳|^採用|展示後モデル|オリジナル展示|使わない|^基準|^修正3" | cut -c1-200
echo "=== 収集(RTモニター)"; grep "完了" /opt/boatrace-rt-monitor/logs/collector.log | tail -3 | cut -c1-100
echo "=== 失敗したサービス"; systemctl list-units --state=failed --no-pager | head -4
echo "=== ミナモのエラー(直近12時間)"; sudo docker logs --since 12h deploy-worker-1 2>&1 | grep -ciE "error|traceback"
echo "=== サイト(200なら正常)"; curl -s -o /dev/null -w "%{http_code}\n" --max-time 15 https://archive.mizu2017boat.com/minamo/
echo "=== バックアップ(RT・直近)"; ls -la /opt/boatrace-rt-monitor/backups | tail -2
