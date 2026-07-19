from boatrace_scraper import parse_profile_html


def test_parse_official_profile_fixture():
    html = """
    <html><head><title>林 祐介（出場予定） ボートレーサー検索へ</title></head>
    <body>
      <div>ハヤシ　ユウスケ</div>
      <dl>
        <dt>登録番号</dt><dd>4214</dd>
        <dt>生年月日</dt><dd>1984/08/08</dd>
        <dt>身長</dt><dd>166cm</dd>
        <dt>体重</dt><dd>58kg</dd>
        <dt>血液型</dt><dd>A型</dd>
        <dt>支部</dt><dd>岡山</dd>
        <dt>出身地</dt><dd>岡山県</dd>
        <dt>登録期</dt><dd>91期</dd>
        <dt>級別</dt><dd>B1級</dd>
      </dl>
      <h2>本日出走予定</h2>
    </body></html>
    """
    racer = parse_profile_html(html, "4214")
    assert racer.registration_number == "4214"
    assert racer.name == "林 祐介"
    assert racer.birth_date == "1984-08-08"
    assert racer.blood_type == "A"
    assert racer.branch == "岡山"
    assert racer.birthplace == "岡山県"
    assert racer.registration_term == "91"
    assert racer.class_level == "B1"
    assert racer.gender == "不明"
