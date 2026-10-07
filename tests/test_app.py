"""录音删段交付 API 的自动化测试。

运行：python -m pytest tests/ -v
"""

import io
import json
import math
import os
import queue
import struct
import tempfile
import threading
import wave
import zipfile

# 必须在 import app 之前指向临时数据库，避免污染工作目录
os.environ["DB_PATH"] = os.path.join(tempfile.mkdtemp(prefix="segdel-import-"), "app.db")

import app as app_module
import pytest

RATE = 16000


def make_wav(seconds=10.0, rate=RATE, channels=1, sampwidth=2, freq=440):
    """生成测试用 WAV 字节。"""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(sampwidth)
        w.setframerate(rate)
        n = int(seconds * rate)
        if sampwidth == 2:
            frames = b"".join(
                struct.pack("<h", int(10000 * math.sin(2 * math.pi * freq * i / rate)))
                for i in range(n)
            )
        else:  # 8 bit 用于构造非法规格
            frames = bytes(i % 256 for i in range(n))
        if channels == 2:
            frames = b"".join(
                struct.pack("<hh", int(5000 * math.sin(2 * math.pi * freq * i / rate)), 0)
                for i in range(n)
            )
        w.writeframes(frames)
    return buf.getvalue()


@pytest.fixture()
def client(tmp_path, monkeypatch):
    db_file = tmp_path / "test.db"
    monkeypatch.setattr(app_module, "DB_PATH", str(db_file))
    app_module.init_db()
    app_module.app.config.update(TESTING=True)
    with app_module.app.test_client() as c:
        yield c


def create_program(client, wav_bytes=None, chapters=None, must_keep=None):
    wav_bytes = wav_bytes if wav_bytes is not None else make_wav()
    data = {"audio": (io.BytesIO(wav_bytes), "test.wav")}
    if chapters is not None:
        data["chapters"] = json.dumps(chapters)
    if must_keep is not None:
        data["must_keep"] = json.dumps(must_keep)
    return client.post("/programs", data=data, content_type="multipart/form-data")


# ---- 创建节目 ---------------------------------------------------------------

def test_create_ok_initial_revision_keeps_everything(client):
    chapters = [{"title": "开场", "start_frame": 0}, {"title": "正文", "start_frame": 80000}]
    must_keep = [{"start_frame": 16000, "end_frame": 32000}]
    resp = create_program(client, chapters=chapters, must_keep=must_keep)
    assert resp.status_code == 201
    body = resp.get_json()
    assert body["revision"] == 0
    assert body["total_frames"] == 10 * RATE
    assert body["keep_segments"] == [{"start_frame": 0, "end_frame": 10 * RATE}]
    assert body["product_frames"] == 10 * RATE
    # 初始版本成品坐标 == 原始坐标
    assert [c["product_start_frame"] for c in body["chapters"]] == [0, 80000]
    assert body["must_keep"] == [must_keep[0]]


def test_create_rejects_bad_wav(client):
    # 非 WAV
    assert create_program(client, wav_bytes=b"not a wav").status_code == 400
    # 立体声
    resp = create_program(client, wav_bytes=make_wav(channels=2))
    assert resp.status_code == 400 and "单声道" in resp.get_json()["message"]
    # 8 kHz
    resp = create_program(client, wav_bytes=make_wav(rate=8000))
    assert resp.status_code == 400 and "16000" in resp.get_json()["message"]
    # 8 位采样
    resp = create_program(client, wav_bytes=make_wav(sampwidth=1))
    assert resp.status_code == 400 and "16" in resp.get_json()["message"]
    # 超过 30 秒
    resp = create_program(client, wav_bytes=make_wav(seconds=31))
    assert resp.status_code == 400 and "30" in resp.get_json()["message"]


def test_create_rejects_bad_chapters_and_must_keep(client):
    resp = create_program(client, chapters=[{"start_frame": 10 * RATE}])  # 越界（右开）
    assert resp.status_code == 400
    resp = create_program(client, chapters=[100, 100])  # 重复
    assert resp.status_code == 400
    resp = create_program(client, must_keep=[{"start_frame": 100, "end_frame": 100}])  # 空区间
    assert resp.status_code == 400
    resp = create_program(client, must_keep=[{"start_frame": 0, "end_frame": 10 * RATE + 1}])
    assert resp.status_code == 400


# ---- 提交保留清单 ------------------------------------------------------------

def _make_program(client):
    chapters = [
        {"title": "c0", "start_frame": 0},
        {"title": "c1", "start_frame": 40000},
        {"title": "c2", "start_frame": 48000},
        {"title": "c3", "start_frame": 80000},
        {"title": "c4", "start_frame": 88000},
        {"title": "c5", "start_frame": 120000},
    ]
    must_keep = [{"start_frame": 50000, "end_frame": 60000}]
    resp = create_program(client, chapters=chapters, must_keep=must_keep)
    assert resp.status_code == 201
    return resp.get_json()["program_id"]


def test_put_keeps_ok_and_projection(client):
    pid = _make_program(client)
    # 保留 [0,32000) [48000,88000) [112000,160000)
    resp = client.put(f"/programs/{pid}/keeps", json={
        "expected_revision": 0,
        "keep_segments": [
            {"start_frame": 0, "end_frame": 32000},
            {"start_frame": 48000, "end_frame": 88000},
            {"start_frame": 112000, "end_frame": 160000},
        ],
    })
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["revision"] == 1
    assert body["product_frames"] == 32000 + 40000 + 48000

    proj = {c["title"]: c["product_start_frame"] for c in body["chapters"]}
    # c0=0 在段首 -> 0；c1=40000 被删；c2=48000 位于保留段起点 -> 保留在 32000；
    # c3=80000 -> 32000+(80000-48000)=64000；c4=88000 是段2右端（右开）且不在段3 -> 删除；
    # c5=120000 -> 32000+40000+(120000-112000)=80000
    assert proj == {"c0": 0, "c2": 32000, "c3": 64000, "c5": 80000}


def test_put_keeps_stale_revision_keeps_old_state(client):
    pid = _make_program(client)
    ok = client.put(f"/programs/{pid}/keeps", json={
        "expected_revision": 0,
        "keep_segments": [{"start_frame": 48000, "end_frame": 88000}],
    })
    assert ok.status_code == 200 and ok.get_json()["revision"] == 1

    # 用过期的 0 再提交 -> 409，说明原因，状态不变
    stale = client.put(f"/programs/{pid}/keeps", json={
        "expected_revision": 0,
        "keep_segments": [{"start_frame": 0, "end_frame": 160000}],
    })
    assert stale.status_code == 409
    body = stale.get_json()
    assert body["error"] == "stale_revision"
    assert body["current_revision"] == 1
    assert "0" in body["message"] and "1" in body["message"]

    cur = client.get(f"/programs/{pid}").get_json()
    assert cur["revision"] == 1
    assert cur["keep_segments"] == [{"start_frame": 48000, "end_frame": 88000}]


def test_put_keeps_invalid_segments_rejected(client):
    pid = _make_program(client)
    cases = [
        [{"start_frame": 100, "end_frame": 100}],                      # 空段
        [{"start_frame": 200, "end_frame": 100}],                      # start > end
        [{"start_frame": -1, "end_frame": 100}],                       # 越界
        [{"start_frame": 0, "end_frame": 160001}],                     # 越界
        [{"start_frame": 50000, "end_frame": 60000},
         {"start_frame": 100, "end_frame": 200}],                      # 未升序
        [{"start_frame": 0, "end_frame": 60000},
         {"start_frame": 59000, "end_frame": 70000}],                  # 重叠
    ]
    for segs in cases:
        resp = client.put(f"/programs/{pid}/keeps",
                          json={"expected_revision": 0, "keep_segments": segs})
        assert resp.status_code == 400, segs
        assert resp.get_json()["message"]
    # 全部失败后修订号不变
    assert client.get(f"/programs/{pid}").get_json()["revision"] == 0


def test_put_keeps_must_keep_must_be_covered(client):
    pid = _make_program(client)
    # 必留口播 [50000,60000) 被切成两段，未完整落入任何保留段
    resp = client.put(f"/programs/{pid}/keeps", json={
        "expected_revision": 0,
        "keep_segments": [
            {"start_frame": 48000, "end_frame": 55000},
            {"start_frame": 55000, "end_frame": 88000},
        ],
    })
    assert resp.status_code == 400
    assert resp.get_json()["error"] == "must_keep_uncovered"
    assert client.get(f"/programs/{pid}").get_json()["revision"] == 0


def test_repeated_edits_use_original_frames_no_drift(client):
    pid = _make_program(client)
    # 第一次：删掉 [32000,48000)
    r1 = client.put(f"/programs/{pid}/keeps", json={
        "expected_revision": 0,
        "keep_segments": [
            {"start_frame": 0, "end_frame": 32000},
            {"start_frame": 48000, "end_frame": 160000},
        ],
    })
    assert r1.status_code == 200
    # 第二次：在原始坐标下进一步删掉 [96000,112000)
    r2 = client.put(f"/programs/{pid}/keeps", json={
        "expected_revision": 1,
        "keep_segments": [
            {"start_frame": 0, "end_frame": 32000},
            {"start_frame": 48000, "end_frame": 96000},
            {"start_frame": 112000, "end_frame": 160000},
        ],
    })
    assert r2.status_code == 200
    body = r2.get_json()
    assert body["revision"] == 2
    # 成品帧 = 32000 + 48000 + 48000 = 128000，与直接按原始帧计算一致
    assert body["product_frames"] == 32000 + 48000 + 48000
    proj = {c["title"]: c["product_start_frame"] for c in body["chapters"]}
    # c5=120000 -> 32000+48000+(120000-112000)=88000
    assert proj["c5"] == 88000
    assert proj["c3"] == 32000 + (80000 - 48000)


# ---- 下载 --------------------------------------------------------------------

def test_download_zip_consistent_with_revision(client):
    pid = _make_program(client)
    client.put(f"/programs/{pid}/keeps", json={
        "expected_revision": 0,
        "keep_segments": [
            {"start_frame": 0, "end_frame": 32000},
            {"start_frame": 48000, "end_frame": 88000},
        ],
    })
    resp = client.get(f"/programs/{pid}/download")
    assert resp.status_code == 200
    assert resp.mimetype == "application/zip"
    zf = zipfile.ZipFile(io.BytesIO(resp.data))
    assert set(zf.namelist()) == {"audio.wav", "chapters.json"}

    # WAV 是真正裁接过的：帧数 == 保留段总长
    with wave.open(io.BytesIO(zf.read("audio.wav")), "rb") as w:
        assert w.getframerate() == RATE
        assert w.getnchannels() == 1
        assert w.getsampwidth() == 2
        assert w.getnframes() == 32000 + 40000

    doc = json.loads(zf.read("chapters.json"))
    assert doc["revision"] == 1
    assert doc["product_frames"] == 32000 + 40000
    starts = {c["title"]: c["start_frame"] for c in doc["chapters"]}
    assert starts == {"c0": 0, "c2": 32000, "c3": 64000}

    # 再次编辑后下载：WAV 与 JSON 同属修订 2
    client.put(f"/programs/{pid}/keeps", json={
        "expected_revision": 1,
        "keep_segments": [{"start_frame": 48000, "end_frame": 88000}],
    })
    resp = client.get(f"/programs/{pid}/download")
    zf = zipfile.ZipFile(io.BytesIO(resp.data))
    doc = json.loads(zf.read("chapters.json"))
    with wave.open(io.BytesIO(zf.read("audio.wav")), "rb") as w:
        wav_frames = w.getnframes()
    assert doc["revision"] == 2
    assert wav_frames == doc["product_frames"] == 40000
    starts = {c["title"]: c["start_frame"] for c in doc["chapters"]}
    assert starts == {"c2": 0, "c3": 32000}


def test_download_wav_content_matches_original_splice(client):
    """裁接结果应与原始音频对应帧完全一致（原始音频不变）。"""
    wav_bytes = make_wav(seconds=2)
    resp = create_program(client, wav_bytes=wav_bytes)
    pid = resp.get_json()["program_id"]
    client.put(f"/programs/{pid}/keeps", json={
        "expected_revision": 0,
        "keep_segments": [
            {"start_frame": 100, "end_frame": 200},
            {"start_frame": 300, "end_frame": 500},
        ],
    })
    zf = zipfile.ZipFile(io.BytesIO(client.get(f"/programs/{pid}/download").data))
    product = zf.read("audio.wav")
    with wave.open(io.BytesIO(wav_bytes), "rb") as w:
        w.setpos(100)
        expect = w.readframes(100)
        w.setpos(300)
        expect += w.readframes(200)
    with wave.open(io.BytesIO(product), "rb") as w:
        actual = w.readframes(w.getnframes())
    assert actual == expect


# ---- 持久化 ------------------------------------------------------------------

def test_program_survives_reconnect(client, tmp_path, monkeypatch):
    """模拟重启：所有状态都在 SQLite 文件中，新连接可继续编辑。"""
    pid = _make_program(client)
    client.put(f"/programs/{pid}/keeps", json={
        "expected_revision": 0,
        "keep_segments": [{"start_frame": 48000, "end_frame": 88000}],
    })
    # 重新指向同一个数据库文件并重新初始化（等价于进程重启后打开已有库）
    db_path = app_module.DB_PATH
    app_module.init_db()
    cur = client.get(f"/programs/{pid}").get_json()
    assert cur["revision"] == 1
    nxt = client.put(f"/programs/{pid}/keeps", json={
        "expected_revision": 1,
        "keep_segments": [{"start_frame": 50000, "end_frame": 60000}],
    })
    assert nxt.status_code == 200 and nxt.get_json()["revision"] == 2


def test_unknown_program_404(client):
    assert client.get("/programs/nope").status_code == 404
    assert client.put("/programs/nope/keeps",
                      json={"expected_revision": 0, "keep_segments": []}).status_code == 404
    assert client.get("/programs/nope/download").status_code == 404


# ---- 旧版建议合入（三方合入）---------------------------------------------------
def _merge(client, pid, base, segments):
    return client.post(f"/programs/{pid}/merges", json={
        "base_revision": base,
        "keep_segments": segments,
    })


def _setup_two_branches(client):
    """基版修订 0 全保留；修订 1 由当前用户删去 [32000,48000)。

    旧版建议仍以修订 0 为依据，可删与之不相交的其它区间。
    """
    pid = _make_program(client)
    r1 = client.put(f"/programs/{pid}/keeps", json={
        "expected_revision": 0,
        "keep_segments": [
            {"start_frame": 0, "end_frame": 32000},
            {"start_frame": 48000, "end_frame": 160000},
        ],
    })
    assert r1.status_code == 200 and r1.get_json()["revision"] == 1
    return pid


def test_merge_disjoint_changes_synthesizes_revision(client):
    pid = _setup_two_branches(client)
    # 建议（依据修订 0）删去 [96000,112000)，与当前已删的 [32000,48000) 互不相交
    resp = _merge(client, pid, 0, [
        {"start_frame": 0, "end_frame": 96000},
        {"start_frame": 112000, "end_frame": 160000},
    ])
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["revision"] == 2
    assert body["keep_segments"] == [
        {"start_frame": 0, "end_frame": 32000},
        {"start_frame": 48000, "end_frame": 96000},
        {"start_frame": 112000, "end_frame": 160000},
    ]
    assert body["product_frames"] == 32000 + 48000 + 48000
    # 章节查询反映合并后的同一修订：c5=120000 落在 [112000,160000)
    proj = {c["title"]: c["product_start_frame"] for c in body["chapters"]}
    assert proj["c5"] == 32000 + 48000 + (120000 - 112000)
    assert "c1" not in proj  # 40000 落在当前侧删除区


def test_merge_seen_after_get_and_download(client):
    pid = _setup_two_branches(client)
    merged_segments = [
        {"start_frame": 0, "end_frame": 32000},
        {"start_frame": 48000, "end_frame": 96000},
        {"start_frame": 112000, "end_frame": 160000},
    ]
    assert _merge(client, pid, 0, [
        {"start_frame": 0, "end_frame": 96000},
        {"start_frame": 112000, "end_frame": 160000},
    ]).status_code == 200

    cur = client.get(f"/programs/{pid}").get_json()
    assert cur["revision"] == 2 and cur["keep_segments"] == merged_segments

    zf = zipfile.ZipFile(io.BytesIO(client.get(f"/programs/{pid}/download").data))
    with wave.open(io.BytesIO(zf.read("audio.wav")), "rb") as w:
        wav_frames = w.getnframes()
    doc = json.loads(zf.read("chapters.json"))
    assert doc["revision"] == 2
    assert wav_frames == doc["product_frames"] == 32000 + 48000 + 48000
    starts = {c["title"]: c["start_frame"] for c in doc["chapters"]}
    assert starts["c5"] == 32000 + 48000 + 8000


def test_merge_overlapping_changes_conflict_409(client):
    pid = _setup_two_branches(client)
    # 建议删 [40000,56000)，与当前删除 [32000,48000) 相交于 [40000,48000)
    resp = _merge(client, pid, 0, [
        {"start_frame": 0, "end_frame": 40000},
        {"start_frame": 56000, "end_frame": 160000},
    ])
    assert resp.status_code == 409
    body = resp.get_json()
    assert body["error"] == "merge_conflict"
    assert body["base_revision"] == 0
    assert body["current_revision"] == 1
    assert body["conflicts"] == [{"start_frame": 40000, "end_frame": 48000}]
    # 当前版本不变
    cur = client.get(f"/programs/{pid}").get_json()
    assert cur["revision"] == 1
    assert cur["keep_segments"] == [
        {"start_frame": 0, "end_frame": 32000},
        {"start_frame": 48000, "end_frame": 160000},
    ]


def test_merge_both_sides_delete_same_segment_is_conflict(client):
    pid = _setup_two_branches(client)
    # 建议清单与当前修订完全相同：双方相对基版都删了 [32000,48000)，不能擅自视为一致
    resp = _merge(client, pid, 0, [
        {"start_frame": 0, "end_frame": 32000},
        {"start_frame": 48000, "end_frame": 160000},
    ])
    assert resp.status_code == 409
    body = resp.get_json()
    assert body["error"] == "merge_conflict"
    assert body["conflicts"] == [{"start_frame": 32000, "end_frame": 48000}]
    assert client.get(f"/programs/{pid}").get_json()["revision"] == 1


def test_merge_adjacent_but_disjoint_changes_ok(client):
    pid = _setup_two_branches(client)
    # 当前删 [32000,48000)；建议删 [16000,32000)——仅相邻、帧区间不相交
    resp = _merge(client, pid, 0, [
        {"start_frame": 0, "end_frame": 16000},
        {"start_frame": 32000, "end_frame": 160000},
    ])
    assert resp.status_code == 200
    assert resp.get_json()["keep_segments"] == [
        {"start_frame": 0, "end_frame": 16000},
        {"start_frame": 48000, "end_frame": 160000},
    ]


def test_merge_base_revision_missing_404(client):
    pid = _setup_two_branches(client)
    resp = _merge(client, pid, 7, [
        {"start_frame": 0, "end_frame": 160000},
    ])
    assert resp.status_code == 404
    assert resp.get_json()["error"] == "base_revision_not_found"
    assert resp.get_json()["current_revision"] == 1
    assert client.get(f"/programs/{pid}").get_json()["revision"] == 1


def test_merge_bad_base_revision_field_400(client):
    pid = _setup_two_branches(client)
    resp = client.post(f"/programs/{pid}/merges", json={
        "base_revision": "0",
        "keep_segments": [{"start_frame": 0, "end_frame": 160000}],
    })
    assert resp.status_code == 400 and resp.get_json()["error"] == "invalid_base_revision"

    resp = client.post(f"/programs/{pid}/merges", json={
        "keep_segments": [{"start_frame": 0, "end_frame": 160000}],
    })
    assert resp.status_code == 400 and resp.get_json()["error"] == "missing_base_revision"
    assert client.get(f"/programs/{pid}").get_json()["revision"] == 1


def test_merge_invalid_suggested_segments_400(client):
    pid = _setup_two_branches(client)
    resp = _merge(client, pid, 0, [
        {"start_frame": 0, "end_frame": 60000},
        {"start_frame": 59000, "end_frame": 70000},  # 重叠、非升序
    ])
    assert resp.status_code == 400 and resp.get_json()["error"] == "invalid_segments"

    resp = _merge(client, pid, 0, [{"start_frame": -1, "end_frame": 100}])
    assert resp.status_code == 400 and resp.get_json()["error"] == "invalid_segments"

    resp = client.post(f"/programs/{pid}/merges",
                       data="not-json", content_type="application/json")
    assert resp.status_code == 400 and resp.get_json()["error"] == "invalid_json"
    assert client.get(f"/programs/{pid}").get_json()["revision"] == 1


def test_merge_result_must_cover_must_keep_400(client):
    pid = _setup_two_branches(client)
    # 必留口播 [50000,60000)；建议删 [48000,96000)，与当前删除不相交，
    # 但合并结果把口播整段删掉 -> 拒绝，当前版本不变
    resp = _merge(client, pid, 0, [
        {"start_frame": 0, "end_frame": 48000},
        {"start_frame": 96000, "end_frame": 160000},
    ])
    assert resp.status_code == 400
    body = resp.get_json()
    assert body["error"] == "must_keep_uncovered"
    assert body["must_keep"] == {"start_frame": 50000, "end_frame": 60000}
    cur = client.get(f"/programs/{pid}").get_json()
    assert cur["revision"] == 1


def test_merge_against_nonzero_base_with_independent_followups(client):
    pid = _setup_two_branches(client)
    # 再以修订 1 为基版：当前侧（修订 2）删 [96000,112000)，建议侧删 [120000,130000)
    assert client.put(f"/programs/{pid}/keeps", json={
        "expected_revision": 1,
        "keep_segments": [
            {"start_frame": 0, "end_frame": 32000},
            {"start_frame": 48000, "end_frame": 96000},
            {"start_frame": 112000, "end_frame": 160000},
        ],
    }).status_code == 200
    resp = _merge(client, pid, 1, [
        {"start_frame": 0, "end_frame": 32000},
        {"start_frame": 48000, "end_frame": 120000},
        {"start_frame": 130000, "end_frame": 160000},
    ])
    assert resp.status_code == 200
    assert resp.get_json()["keep_segments"] == [
        {"start_frame": 0, "end_frame": 32000},
        {"start_frame": 48000, "end_frame": 96000},
        {"start_frame": 112000, "end_frame": 120000},
        {"start_frame": 130000, "end_frame": 160000},
    ]


def test_merge_unknown_program_404(client):
    assert client.post("/programs/nope/merges", json={
        "base_revision": 0,
        "keep_segments": [{"start_frame": 0, "end_frame": 100}],
    }).status_code == 404


def test_concurrent_merges_both_take_effect(client):
    """两个并发合入都基于修订 0、删除互不相交的区间：串行化后两处删除都生效。"""
    pid = _make_program(client)  # 必留口播 [50000,60000)
    barrier = threading.Barrier(2)
    results = {}

    def merge_a():
        with app_module.app.test_client() as c:
            barrier.wait()
            results["a"] = _merge(c, pid, 0, [
                {"start_frame": 0, "end_frame": 16000},
                {"start_frame": 32000, "end_frame": 160000},
            ])

    def merge_b():
        with app_module.app.test_client() as c:
            barrier.wait()
            results["b"] = _merge(c, pid, 0, [
                {"start_frame": 0, "end_frame": 96000},
                {"start_frame": 112000, "end_frame": 160000},
            ])

    t1, t2 = threading.Thread(target=merge_a), threading.Thread(target=merge_b)
    t1.start(); t2.start(); t1.join(); t2.join()

    assert results["a"].status_code == 200, results["a"].status
    assert results["b"].status_code == 200, results["b"].status
    assert {results["a"].get_json()["revision"],
            results["b"].get_json()["revision"]} == {1, 2}

    cur = client.get(f"/programs/{pid}").get_json()
    assert cur["revision"] == 2
    # 两处删除 [16000,32000) 与 [96000,112000) 都在最终修订中
    assert cur["keep_segments"] == [
        {"start_frame": 0, "end_frame": 16000},
        {"start_frame": 32000, "end_frame": 96000},
        {"start_frame": 112000, "end_frame": 160000},
    ]
    assert cur["product_frames"] == 16000 + 64000 + 48000


def test_concurrent_merge_response_matches_own_written_revision(client, monkeypatch):
    """回归：第二笔合入在第一笔提交后、响应前完成时，两笔的成功响应都必须
    对应各自写入的修订（修订号、保留段、成品帧数、章节投影），
    第一笔不能返回第二笔写入的修订内容；最终当前版本包含两笔非冲突修改。

    确定性交错的做法：钩住合入处理对 _fetch_program 的两次调用——
    第一次在事务内（已持有写锁），第二次在 COMMIT 之后、响应之前。
    先让第一笔进入事务再启动第二笔，保证提交次序；第一笔在第二次调用处
    卡住，直到第二笔的请求完整结束才放行去组装自己的响应。
    """
    pid = _make_program(client)  # 修订 0 全保留，必留口播 [50000,60000)

    orig_fetch = app_module._fetch_program
    entered = queue.Queue()               # 某线程的合入事务已拿到写锁
    first_committed = threading.Event()   # 第一笔已提交并停在响应前
    second_finished = threading.Event()   # 第二笔请求已完整结束
    calls = {}
    gate_lock = threading.Lock()
    gate = {"taken": False}

    def spy_fetch(db, pid_arg):
        ident = threading.get_ident()
        calls[ident] = calls.get(ident, 0) + 1
        n = calls[ident]
        if n == 1:
            entered.put(ident)
        elif n == 2:
            # “COMMIT 之后、响应之前”：只卡住先到者（即第一笔），
            # 等第二笔彻底完成，确定地复现响应读到他人修订的交错
            with gate_lock:
                is_first = not gate["taken"]
                gate["taken"] = True
            if is_first:
                first_committed.set()
                assert second_finished.wait(timeout=10)
        return orig_fetch(db, pid_arg)

    monkeypatch.setattr(app_module, "_fetch_program", spy_fetch)

    results = {}

    def merge_first():
        # 删 [16000,32000)：先进入事务，保证先提交（产生修订 1）
        with app_module.app.test_client() as c:
            results["first"] = _merge(c, pid, 0, [
                {"start_frame": 0, "end_frame": 16000},
                {"start_frame": 32000, "end_frame": 160000},
            ])

    def merge_second():
        # 删 [96000,112000)：与第一笔不相交，合入后产生修订 2
        with app_module.app.test_client() as c:
            results["second"] = _merge(c, pid, 0, [
                {"start_frame": 0, "end_frame": 96000},
                {"start_frame": 112000, "end_frame": 160000},
            ])
        second_finished.set()

    t_first = threading.Thread(target=merge_first)
    t_first.start()
    entered.get(timeout=10)  # 第一笔已持有写锁，第二笔必然排在其后提交
    t_second = threading.Thread(target=merge_second)
    t_second.start()
    t_first.join(timeout=15)
    t_second.join(timeout=15)
    assert not t_first.is_alive() and not t_second.is_alive()
    # 确认目标交错真的发生：第一笔提交后曾停在响应前等第二笔完成
    assert first_committed.is_set()

    resp_first = results["first"]
    resp_second = results["second"]
    assert resp_first.status_code == 200, resp_first.get_json()
    assert resp_second.status_code == 200, resp_second.get_json()
    first = resp_first.get_json()
    second = resp_second.get_json()

    # 第一笔的响应必须是它自己写入的修订 1：只有自己那处删除
    assert first["revision"] == 1
    assert first["keep_segments"] == [
        {"start_frame": 0, "end_frame": 16000},
        {"start_frame": 32000, "end_frame": 160000},
    ]
    assert first["product_frames"] == 16000 + 128000
    proj_first = {c["title"]: c["product_start_frame"] for c in first["chapters"]}
    assert proj_first["c5"] == 16000 + (120000 - 32000)

    # 第二笔的响应是它自己写入的修订 2：两处删除都在
    assert second["revision"] == 2
    assert second["keep_segments"] == [
        {"start_frame": 0, "end_frame": 16000},
        {"start_frame": 32000, "end_frame": 96000},
        {"start_frame": 112000, "end_frame": 160000},
    ]
    assert second["product_frames"] == 16000 + 64000 + 48000
    proj_second = {c["title"]: c["product_start_frame"] for c in second["chapters"]}
    assert proj_second["c5"] == 16000 + 64000 + (120000 - 112000)

    # 最终当前版本仍是修订 2：两笔非冲突修改都保留
    cur = client.get(f"/programs/{pid}").get_json()
    assert cur["revision"] == 2
    assert cur["keep_segments"] == second["keep_segments"]
