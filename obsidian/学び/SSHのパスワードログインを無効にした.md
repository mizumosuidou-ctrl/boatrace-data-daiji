---
tags: [学び]
日付: 2026-10-09
分類: サーバー
---
# SSH のパスワードログインを無効にした

## 分かったこと（読むだけの下調べ）
- パスワードでのログインが有効で（`PasswordAuthentication yes`。クラウドの設定ファイル `50-cloud-init.conf` が書いていた）、`ubuntu` のパスワードも設定されていた。
- 1日で **約7,600回** の「パスワードでの侵入の試み」（`Failed password`）があった。世界中のボットが総当たりをしている。
- 普段のログインは最初から鍵だけを使っていて、パスワードでは入っていなかった（Mac 以外から入ることも無い）。鍵は3個登録されていた（持ち主の確認は、下の「次にやること」）。

## やったこと
- `/etc/ssh/sshd_config.d/00-no-password.conf` に `PasswordAuthentication no` を1行。
  設定ファイルは名前順に読まれ、**先に読んだ値が優先**されるので、`50-cloud-init.conf`（yes）より前の `00-` にした。
- 反映の前に `sudo sshd -t` で文法を検査し、`reload`（今つながっている接続は切れない）。`sshd -T` で `passwordauthentication no` を確認。
- ついでに、間違えて作った重複フォルダ（サーバー上の `~/Desktop/boatrace-backup-20261008`、599MB）を、元のバックアップと `cmp` で1バイトも違わないと確認してから削除。

## 戻し方（鍵で入れなくなったとき）
- つながっている窓で：`sudo rm /etc/ssh/sshd_config.d/00-no-password.conf && sudo systemctl reload ssh`
- 窓を閉じてしまった場合：KAGOYA のコントロールパネルのコンソールから入って、同じコマンド。

## 次にやること
- 登録されている鍵3個の持ち主を確かめる（`ssh-keygen -lf ~/.ssh/authorized_keys` で指紋とコメントが出る）。知らない鍵があれば消す
- 試みのログが多いのは変わらない（入れないだけ）。気になるなら fail2ban を入れる（必須ではない）
