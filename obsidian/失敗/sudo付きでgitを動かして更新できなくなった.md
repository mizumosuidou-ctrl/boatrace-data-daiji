---
tags: [失敗]
日付: 2026-10-09
分類: サーバー
---
# sudo 付きで git を動かして、サーバーの更新ができなくなった

## 起きたこと
- `update.sh` が `error: insufficient permission for adding an object to repository database .git/objects` で止まった。サーバーのコードは更新されず、本番は古いコードのまま動き続けた（影響なし）。
- `/opt/minamo` の中で、持ち主が root のファイルが **約1,700個**（`.git` の中に 286 個、ミナモのプログラムのファイルにも）。

## 原因
- どこかで **`sudo` を付けて git を動かした**（更新・取り込みなど）。root が作ったファイルは、ubuntu では上書きも追加もできない。
- 直前の更新（#184）は通っていたので、その後に起きた。

## 直したこと
- 許可をもらって、`/opt/minamo` の持ち主を ubuntu に戻した（`/opt/minamo/var` は学習の結果・データなのでわざと root のまま、触らない）：
  `sudo find /opt/minamo -path /opt/minamo/var -prune -o ! -user ubuntu -exec chown ubuntu:ubuntu {} +`
- `deploy/update.sh` の最初に、`.git` に持ち主のずれたファイルが無いか確かめて、あれば上のコマンドを表示して止まるようにした。

## 学び（次からどうするか）
- 更新は **`sudo` を付けずに** `bash /opt/minamo/deploy/update.sh`（必要なところだけ、中で sudo を使う）
- 直した直後でも 1 個だけ持ち主のずれたファイルが残った（cron の root の処理が、直す間に作ったものと思われる）。更新は問題なく通った
- → [[失敗/Macとサーバーの画面を取り違えた]]（サーバーの作業は、画面と権限を確かめてから）
