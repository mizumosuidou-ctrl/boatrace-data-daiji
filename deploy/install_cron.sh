#!/usr/bin/env bash
# 学習の自動実行と、レースタイムモニターの予想の書き出しを登録する（/etc/cron.d/minamo を書く）。もう一度実行しても同じ内容で上書きするだけ。
#   使い方:  sudo bash /opt/minamo/deploy/install_cron.sh
#   やめる:  sudo rm /etc/cron.d/minamo
set -euo pipefail
chmod +x /opt/minamo/deploy/cron_ml.sh /opt/minamo/deploy/backup.sh /opt/minamo/deploy/rtm_live.sh
cat > /etc/cron.d/minamo <<'CRON'
# MINAMO の学習の自動実行（deploy/install_cron.sh が書いたもの）。時刻はサーバーの時計（日本時間）
SHELL=/bin/bash
PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
# 火〜日の3時10分：前日の実績を足して学習し直す
10 3 * * 0,2-6 root /opt/minamo/deploy/cron_ml.sh daily
# 月曜の3時10分：データベースからの書き出しからやり直す（コードも更新）
10 3 * * 1 root /opt/minamo/deploy/cron_ml.sh weekly
# 月曜の2時40分：作り直せない記録のバックアップ（/opt/minamo/backup、直近4週）
40 2 * * 1 root /opt/minamo/deploy/backup.sh >> /opt/minamo/var/cron_backup.log 2>&1
# 月曜の8時5分：週報（先週の実戦の成績と学習の様子）を Discord に送る
5 8 * * 1 root cd /opt/minamo/deploy && docker compose run --rm worker python -m minamo weekly --send >> /opt/minamo/var/cron_weekly.log 2>&1
# 8時〜23時台の毎分（ミッドナイト開催の最終Rは22時台後半まで）：レースタイムモニターの今日の予想（1番手）を書き出す（試し買い「一致」が締切前に読む。データベースは読むだけ）
* 8-23 * * * root /opt/minamo/deploy/rtm_live.sh >> /opt/minamo/var/cron_rtm_live.log 2>&1
# 10分おき：見張り（コンテナ・サイト・一覧の更新・RTモニターの収集・学習・ディスク・バックアップ）。異常のときだけ Discord に送る（読むだけ）
*/10 * * * * root /usr/bin/python3 /opt/minamo/minamo/watchdog.py >> /opt/minamo/var/cron_watchdog.log 2>&1
CRON
chmod 644 /etc/cron.d/minamo
echo "登録しました（サーバーの時刻 $(date '+%F %H:%M %Z')）:"
grep -v '^#' /etc/cron.d/minamo | grep -v '^$'
