#!/usr/bin/env bash
# 学習の自動実行（install_cron.sh が /etc/cron.d/minamo に登録する）。データベースは読むだけ。
#   daily  … 前日までの実績を公式サイトで足す（ページ・ダウンロードデータ）→ 学習し直す → 買い目の比べ（補正の値も更新）→ 予想の仕組みを入れ替える
#   weekly … データベースからの書き出しからやり直す（ml_refresh.sh）。最新のコードに更新してから
# 2つが重なったら、あとから始まった方は休む。記録は /opt/minamo/var/cron_<daily|weekly>.log
set -uo pipefail
cd "$(dirname "$0")"
mode=${1:-daily}
log=/opt/minamo/var/cron_$mode.log
if [ -f "$log" ] && [ "$(stat -c %s "$log")" -gt 2000000 ]; then mv -f "$log" "$log.old"; fi
exec >>"$log" 2>&1
echo "=== $(date '+%F %T') $mode 始め ==="
exec 9>/tmp/minamo-ml.lock
if ! flock -n 9; then
  echo "ほかの学習が動いているので、今回は休みます"
  exit 0
fi
status=0
case "$mode" in
  daily)
    docker compose run --rm worker python -m minamo ml-facts || echo "（実績を足せませんでした。続けます）"
    docker compose run --rm worker python -m minamo ml-official --fan || echo "（公式のダウンロードデータを取り込めませんでした。続けます）"
    docker compose run --rm worker python -m minamo ml-original-live || echo "（本番のオリジナル展示を書き出せませんでした。続けます）"
    docker compose run --rm worker python -m minamo ml-train && \
      { docker compose run --rm worker python -m minamo ev-check > /dev/null || echo "（買い目の比べに失敗。続けます）"; } && \
      docker compose up -d worker || status=$?
    ;;
  weekly)
    git -C /opt/minamo pull -q || echo "（コードを更新できませんでした。今のコードで続けます）"
    docker compose build -q worker && bash ./ml_refresh.sh && docker compose up -d --build worker || status=$?
    ;;
  *)
    echo "使い方: cron_ml.sh daily|weekly"; status=2 ;;
esac
echo "=== $(date '+%F %T') $mode 終わり（status=$status）==="
exit $status
