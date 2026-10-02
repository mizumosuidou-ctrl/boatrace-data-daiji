"""boatrace.jp のページ構造を模したテスト用HTML。"""

RACERS = [
    (1, "4320", "A1", "峰　　竜太", "佐賀/佐賀", "39", "51.0", "F0", "L0", "0.14", ("7.85", "62.50", "80.21"), ("8.10", "66.67", "83.33"), ("34", "45.10", "60.78"), ("56", "32.00", "48.00")),
    (2, "4444", "A2", "桐生　順平", "埼玉/福島", "37", "52.0", "F1", "L0", "0.16", ("6.12", "41.00", "60.00"), ("0.00", "0.00", "0.00"), ("12", "30.00", "44.00"), ("22", "35.00", "51.00")),
    (3, "5001", "B1", "新人　太郎", "東京/東京", "24", "53.5", "F0", "L0", "0.18", ("4.50", "22.00", "38.00"), ("4.20", "20.00", "35.00"), ("45", "38.00", "55.00"), ("18", "30.00", "45.00")),
    (4, "3960", "A1", "菊地　孝平", "静岡/静岡", "46", "52.5", "F0", "L0", "0.12", ("7.20", "55.00", "70.00"), ("7.00", "50.00", "65.00"), ("61", "52.00", "68.00"), ("40", "36.00", "52.00")),
    (5, "4800", "B1", "中堅　一郎", "福岡/福岡", "31", "54.0", "F0", "L0", "0.17", ("5.10", "30.00", "45.00"), ("5.50", "33.00", "48.00"), ("27", "28.00", "40.00"), ("33", "31.00", "47.00")),
    (6, "5200", "B2", "若手　二郎", "大阪/大阪", "22", "50.0", "F0", "L0", "0.19", ("2.80", "10.00", "20.00"), ("0.00", "0.00", "0.00"), ("50", "34.00", "50.00"), ("61", "29.00", "44.00")),
]

ZEN = "０１２３４５６"


def _racer_tbody(r):
    boat, toban, grade, name, branch, age, weight, f, l, st, nat, loc, mot, bt = r
    return f"""
<tbody class="is-fs12 ">
<tr>
  <td class="is-boatColor{boat} is-fs14" rowspan="4">{ZEN[boat]}</td>
  <td rowspan="4"><a href="/owpc/pc/data/racersearch/profile?toban={toban}"><img src="x.jpg"></a></td>
  <td rowspan="4">
    <div class="is-fs11">{toban}<span> / </span><span class="is-fColor1">{grade}</span></div>
    <div class="is-fs18 is-fBold"><a href="/owpc/pc/data/racersearch/profile?toban={toban}">{name}</a></div>
    <div class="is-fs11">{branch}<br>{age}歳/{weight}kg</div>
  </td>
  <td class="is-lineH2" rowspan="4">{f}<br>{l}<br>{st}</td>
  <td class="is-lineH2" rowspan="4">{nat[0]}<br>{nat[1]}<br>{nat[2]}</td>
  <td class="is-lineH2" rowspan="4">{loc[0]}<br>{loc[1]}<br>{loc[2]}</td>
  <td class="is-lineH2" rowspan="4">{mot[0]}<br>{mot[1]}<br>{mot[2]}</td>
  <td class="is-lineH2" rowspan="4">{bt[0]}<br>{bt[1]}<br>{bt[2]}</td>
  <td rowspan="4">&nbsp;</td>
  <td>1</td><td>3</td>
</tr>
<tr><td>{boat}</td><td>2</td></tr>
<tr><td>.15</td><td>.12</td></tr>
<tr><td>2</td><td>1</td></tr>
</tbody>"""


RACELIST_HTML = f"""<html><head><title>出走表</title></head><body>
<div class="heading2_area"><img src="/static_extra/pc/images/text_place2_12.png" alt="住之江"></div>
<div class="heading2_title"><h2 class="heading2_titleName">第54回 高松宮記念特選競走</h2></div>
<div class="table1 h-mt10"><table><tbody><tr>
  <th>締切予定時刻</th>
  <td>15:17</td><td>15:44</td><td>16:11</td><td>16:39</td><td>17:08</td><td>17:37</td>
  <td>18:07</td><td>18:37</td><td>19:08</td><td>19:39</td><td>20:11</td><td class="is-fBold">20:45</td>
</tr></tbody></table></div>
<div class="title16_titleDetail__add2020"><h3 class="title16_titleDetail__add2020">予選　　　　1800m　安定板使用</h3></div>
<div class="table1 is-tableFixed__3rdadd"><table>
<thead><tr><th>枠</th><th>写真</th><th>登録番号/級別</th><th>F数 L数 平均ST</th><th>全国</th><th>当地</th><th>モーター</th><th>ボート</th></tr></thead>
{''.join(_racer_tbody(r) for r in RACERS)}
</table></div></body></html>"""


def _before_tbody(boat, toban, weight, exh, tilt):
    return f"""
<tbody class="is-fs12">
<tr>
  <td class="is-boatColor{boat}" rowspan="4">{boat}</td>
  <td rowspan="4"><img src="x.jpg"></td>
  <td rowspan="2"><a href="/owpc/pc/data/racersearch/profile?toban={toban}">選手</a></td>
  <td rowspan="2">{weight}kg</td>
  <td rowspan="4">{exh}</td>
  <td rowspan="4">{tilt}</td>
  <td rowspan="4">&nbsp;</td>
  <td rowspan="4">&nbsp;</td>
  <td>R</td><td>1</td>
</tr>
<tr><td>進入</td><td>1</td></tr>
<tr><td>0.0</td><td>ST</td></tr>
<tr><td>成績</td><td>1</td></tr>
</tbody>"""


BEFORE_ROWS = [(1, "4320", "51.0", "6.72", "-0.5"), (2, "4444", "52.0", "6.80", "0.0"), (3, "5001", "53.5", "6.85", "-0.5"),
               (4, "3960", "52.5", "6.70", "0.5"), (5, "4800", "54.0", "6.83", "-0.5"), (6, "5200", "50.0", "6.90", "-0.5")]
# 進入：4号艇が3コースへ前付け
START_ORDER = [(1, ".12"), (2, ".15"), (4, ".08"), (3, "F.02"), (5, ".17"), (6, ".20")]

BEFOREINFO_HTML = f"""<html><body>
<div class="table1"><table>
<thead><tr><th>枠</th><th>写真</th><th>ボートレーサー</th><th>体重</th><th>展示タイム</th><th>チルト</th></tr></thead>
{''.join(_before_tbody(*r) for r in BEFORE_ROWS)}
</table></div>
<div class="table1"><table class="is-w238"><tbody>
{''.join(f'<tr><td><div class="table1_boatImage1"><span class="table1_boatImage1Number is-type{b}">{b}</span><span class="table1_boatImage1Time">{t}</span></div></td></tr>' for b, t in START_ORDER)}
</tbody></table></div>
<div class="weather1">
  <div class="weather1_body">
    <div class="weather1_bodyUnit is-direction"><p class="weather1_bodyUnitImage is-direction7"></p><div class="weather1_bodyUnitLabel"><span class="weather1_bodyUnitLabelTitle">気温</span><span class="weather1_bodyUnitLabelData">24.0℃</span></div></div>
    <div class="weather1_bodyUnit is-weather"><p class="weather1_bodyUnitImage is-weather1"></p><div class="weather1_bodyUnitLabel"><span class="weather1_bodyUnitLabelTitle">晴</span></div></div>
    <div class="weather1_bodyUnit is-wind"><div class="weather1_bodyUnitLabel"><span class="weather1_bodyUnitLabelTitle">風速</span><span class="weather1_bodyUnitLabelData">6m</span></div></div>
    <div class="weather1_bodyUnit is-windDirection"><p class="weather1_bodyUnitImage is-wind13"></p></div>
    <div class="weather1_bodyUnit is-waterTemperature"><div class="weather1_bodyUnitLabel"><span class="weather1_bodyUnitLabelTitle">水温</span><span class="weather1_bodyUnitLabelData">22.0℃</span></div></div>
    <div class="weather1_bodyUnit is-wave"><div class="weather1_bodyUnitLabel"><span class="weather1_bodyUnitLabelTitle">波高</span><span class="weather1_bodyUnitLabelData">5cm</span></div></div>
  </div>
</div></body></html>"""


def odds_html():
    from minamo.parsers import trifecta_order

    order = trifecta_order()
    value = {c: float(10 + i) for i, c in enumerate(sorted(order))}
    rows = []
    for r in range(20):
        cells = []
        for f in range(6):
            combo = order[r * 6 + f]
            s, t = combo.split("-")[1:]
            if r % 4 == 0:
                cells.append(f'<td class="is-boatColor{s}" rowspan="4">{s}</td>')
            cells.append(f'<td class="is-boatColor{t}">{t}</td><td class="oddsPoint">{value[combo]}</td>')
        rows.append("<tr>" + "".join(cells) + "</tr>")
    return "<table><tbody>" + "".join(rows) + "</tbody></table>", value


RESULT_HTML = """<html><body>
<div class="table1"><table class="is-w495">
<thead><tr><th>着</th><th>枠</th><th>ボートレーサー</th><th>レースタイム</th></tr></thead>
<tbody><tr><td class="is-fs14">１</td><td class="is-fs14 is-boatColor4">4</td><td><span class="is-fs12">3960</span> <span class="is-fs18 is-fBold">菊地　孝平</span></td><td>1'48"9</td></tr></tbody>
<tbody><tr><td class="is-fs14">２</td><td class="is-fs14 is-boatColor1">1</td><td><span class="is-fs12">4320</span> <span class="is-fs18 is-fBold">峰　竜太</span></td><td>1'50"1</td></tr></tbody>
<tbody><tr><td class="is-fs14">３</td><td class="is-fs14 is-boatColor2">2</td><td><span class="is-fs12">4444</span> <span class="is-fs18 is-fBold">桐生　順平</span></td><td>1'51"3</td></tr></tbody>
<tbody><tr><td class="is-fs14">４</td><td class="is-fs14 is-boatColor5">5</td><td><span class="is-fs12">4800</span> <span class="is-fs18 is-fBold">中堅　一郎</span></td><td>1'52"0</td></tr></tbody>
<tbody><tr><td class="is-fs14">５</td><td class="is-fs14 is-boatColor6">6</td><td><span class="is-fs12">5200</span> <span class="is-fs18 is-fBold">若手　二郎</span></td><td>1'53"4</td></tr></tbody>
<tbody><tr><td class="is-fs14">Ｆ</td><td class="is-fs14 is-boatColor3">3</td><td><span class="is-fs12">5001</span> <span class="is-fs18 is-fBold">新人　太郎</span></td><td>&nbsp;</td></tr></tbody>
</table></div>
<div class="table1"><table><tbody>
<tr><td><div class="table1_boatImage1"><span class="table1_boatImage1Number is-type1">1</span><span class="table1_boatImage1TimeInner">.11</span></div></td></tr>
<tr><td><div class="table1_boatImage1"><span class="table1_boatImage1Number is-type2">2</span><span class="table1_boatImage1TimeInner">.14</span></div></td></tr>
<tr><td><div class="table1_boatImage1"><span class="table1_boatImage1Number is-type4">4</span><span class="table1_boatImage1TimeInner">.06　まくり</span></div></td></tr>
<tr><td><div class="table1_boatImage1"><span class="table1_boatImage1Number is-type3">3</span><span class="table1_boatImage1TimeInner">F.01</span></div></td></tr>
<tr><td><div class="table1_boatImage1"><span class="table1_boatImage1Number is-type5">5</span><span class="table1_boatImage1TimeInner">.16</span></div></td></tr>
<tr><td><div class="table1_boatImage1"><span class="table1_boatImage1Number is-type6">6</span><span class="table1_boatImage1TimeInner">.19</span></div></td></tr>
</tbody></table></div>
<div class="table1"><table><tbody>
<tr><td rowspan="2">3連単</td><td><div class="numberSet1_row"><span class="numberSet1_number is-type4">4</span><span class="numberSet1_text">-</span><span class="numberSet1_number is-type1">1</span><span class="numberSet1_text">-</span><span class="numberSet1_number is-type2">2</span></div></td><td><span class="is-payout1">¥4,560</span></td><td>15</td></tr>
<tr><td>2連単</td><td><div class="numberSet1_row"><span>4</span><span>-</span><span>1</span></div></td><td><span class="is-payout1">¥1,230</span></td><td>5</td></tr>
</tbody></table></div>
<div class="table1"><table><tbody><tr><th>決まり手</th></tr><tr><td class="is-fs16">まくり</td></tr></tbody></table></div>
</body></html>"""

INDEX_HTML = """<html><body><div class="table1"><table>
<tbody><tr>
  <td class="is-arrow1 is-fBold is-fs15"><a href="/owpc/pc/race/raceindex?jcd=12&hd=20261001"><img alt="住之江"></a></td>
  <td class="is-alignL is-fs11"><a href="/owpc/pc/race/racelist?rno=1&jcd=12&hd=20261001">1R 15:17</a></td>
  <td class="is-alignL is-fBold is-p10-7 is-G1b"><a href="/owpc/pc/race/raceindex?jcd=12&hd=20261001">第54回 高松宮記念特選競走</a></td>
  <td class="is-nighter">3日目</td>
</tr></tbody>
<tbody><tr>
  <td><a href="/owpc/pc/race/raceindex?jcd=24&hd=20261001"><img alt="大村"></a></td>
  <td><a href="/owpc/pc/race/racelist?rno=1&jcd=24&hd=20261001">1R 10:45</a></td>
  <td class="is-alignL is-fBold is-ippan"><a href="/owpc/pc/race/raceindex?jcd=24&hd=20261001">大村市長杯争奪戦</a></td>
  <td>初日</td>
</tr></tbody>
</table></div></body></html>"""


def odds2_html():
    """公式2連単オッズ表と同じ並び（5行×1着6列、そのあとに2連複15）。値は 1-2=1.5, 1-3=1.6 … のように決まる。"""
    from minamo.parsers import exacta_order

    value = {c: 1.5 + i / 10 for i, c in enumerate(sorted(exacta_order()))}
    rows = []
    for r in range(5):
        cells = []
        for f in range(6):
            combo = exacta_order()[r * 6 + f]
            cells.append(f'<td class="is-boatColor{combo[-1]}">{combo[-1]}</td><td class="oddsPoint">{value[combo]:.1f}</td>')
        rows.append("<tr>" + "".join(cells) + "</tr>")
    quinella = "".join('<td class="oddsPoint">9.9</td>' for _ in range(15))
    return f"<html><table><tbody>{''.join(rows)}</tbody></table><table><tr>{quinella}</tr></table></html>", value
