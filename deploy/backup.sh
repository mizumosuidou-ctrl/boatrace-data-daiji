#!/usr/bin/env bash
# 作り直せない記録を1つのファイルに固める（install_cron.sh が毎週登録）。データベースは触らない。
#   サイトの予想・成績（web/data）、MINAMOが記録したオッズ・アビリティなど（var/state）、
#   公式サイトから足した実績・展示・風、補正の値、設定（deploy/.env。鍵が入るので人に渡さない）
# 置き場所: /opt/minamo/backup/minamo-YYYYMMDD.tar.gz（直近 KEEP 個を残す）と、最新への minamo-latest.tar.gz
set -euo pipefail
ROOT=/opt/minamo
DEST=$ROOT/backup
KEEP=${KEEP:-4}
mkdir -p "$DEST"
name="minamo-$(date +%Y%m%d).tar.gz"
cd "$ROOT"
items=(web/data var/state deploy/.env)
for f in var/ml/raw/facts_backfill.csv var/ml/raw/facts_backfill_days.txt var/ml/raw/exhibition_backfill.csv \
         var/ml/raw/weather_backfill.csv var/ml/raw/original.csv var/ml/ev_calib.json; do
  [ -e "$f" ] && items+=("$f")
done
tar czf "$DEST/$name.tmp" "${items[@]}"
mv -f "$DEST/$name.tmp" "$DEST/$name"
chmod 600 "$DEST/$name"
ln -sfn "$name" "$DEST/minamo-latest.tar.gz"
ls -1t "$DEST"/minamo-2*.tar.gz | tail -n +$((KEEP + 1)) | xargs -r rm -f
echo "バックアップ: $DEST/$name（$(du -h "$DEST/$name" | cut -f1)）"
