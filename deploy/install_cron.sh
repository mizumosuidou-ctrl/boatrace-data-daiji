#!/usr/bin/env bash
# 学習の自動実行を登録する（/etc/cron.d/minamo を書く）。もう一度実行しても同じ内容で上書きするだけ。
#   使い方:  sudo bash /opt/minamo/deploy/install_cron.sh
#   やめる:  sudo rm /etc/cron.d/minamo
set -euo pipefail
chmod +x /opt/minamo/deploy/cron_ml.sh /opt/minamo/deploy/backup.sh
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
CRON
chmod 644 /etc/cron.d/minamo
echo "登録しました（サーバーの時刻 $(date '+%F %H:%M %Z')）:"
grep -v '^#' /etc/cron.d/minamo | grep -v '^$'
