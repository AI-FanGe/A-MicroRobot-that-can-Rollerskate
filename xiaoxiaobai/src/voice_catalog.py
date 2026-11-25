# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright 2026 AI-FanGe

"""Voices the robot can speak with, shared by voice_chat.py and the robot API.

Three engines: the Qwen-Omni model's own voices (lowest latency), Qwen3-TTS-Realtime voices,
and Volcengine Doubao TTS voices (needs a Volcengine API key; spoken through the Pi).
"""

from __future__ import annotations

# Every one of these is also in voice_chat.OMNI_VOICES (the Omni model speaks them itself).
QWEN_OMNI = [
    ("Tina", "Tina"), ("Momo", "Momo"), ("Cindy", "Cindy"), ("Cici", "Cici"), ("Mia", "Mia"), ("Chloe", "Chloe"),
    ("Serena", "Serena 苏瑶"), ("Ethan", "Ethan 晨煦"), ("Jennifer", "Jennifer"), ("Ryan", "Ryan 甜茶"),
    ("Dylan", "Dylan 北京话"), ("Sunny", "Sunny 四川话"), ("Eric", "Eric 四川话"), ("Li", "Li 南京话"),
    ("Marcus", "Marcus 陕西话"), ("Peter", "Peter 天津话"), ("Kiki", "Kiki 粤语"), ("Rocky", "Rocky 粤语"),
]
QWEN_TTS = [
    ("Mochi", "Mochi（默认）"), ("Cherry", "Cherry 芊悦"), ("Chelsie", "Chelsie 千雪"), ("Jada", "Jada 上海话"),
    ("Nofish", "Nofish 不吃鱼"), ("Elias", "Elias 墨讲师"),
]
# Doubao TTS 2.0 voices (X-Api-Resource-Id seed-tts-2.0), from the Volcengine voice list.
VOLC_GROUPS = [
    ("通用女声", [
        ("zh_female_vv_uranus_bigtts", "Vivi"), ("zh_female_xiaohe_uranus_bigtts", "小何"),
        ("zh_female_tianmeixiaoyuan_uranus_bigtts", "甜美小源"), ("zh_female_shuangkuaisisi_uranus_bigtts", "爽快思思"),
        ("zh_female_linjianvhai_uranus_bigtts", "邻家女孩"), ("zh_female_kailangjiejie_uranus_bigtts", "开朗姐姐"),
        ("zh_female_qingxinnvsheng_uranus_bigtts", "清新女声"), ("zh_female_cancan_uranus_bigtts", "知性灿灿"),
        ("zh_female_wenroushunv_uranus_bigtts", "温柔淑女"), ("zh_female_tiexinnvsheng_uranus_bigtts", "贴心女声"),
        ("zh_female_qiaopinv_uranus_bigtts", "俏皮女声"), ("zh_female_wenroumama_uranus_bigtts", "温柔妈妈"),
    ]),
    ("通用男声", [
        ("zh_male_m191_uranus_bigtts", "云舟"), ("zh_male_taocheng_uranus_bigtts", "小天"),
        ("zh_male_shaonianzixin_uranus_bigtts", "少年梓辛"), ("zh_male_wennuanahu_uranus_bigtts", "温暖阿虎"),
        ("zh_male_linjiananhai_uranus_bigtts", "邻家男孩"), ("zh_male_huolixiaoge_uranus_bigtts", "活力小哥"),
        ("zh_male_kailangdidi_uranus_bigtts", "开朗弟弟"), ("zh_male_yangguangqingnian_uranus_bigtts", "阳光青年"),
        ("zh_male_yuanboxiaoshu_uranus_bigtts", "渊博小叔"), ("zh_male_ruyaqingnian_uranus_bigtts", "儒雅青年"),
    ]),
    ("童声与萌系", [
        ("zh_male_naiqimengwa_uranus_bigtts", "奶气萌娃"), ("zh_male_liangsangmengzai_uranus_bigtts", "亮嗓萌仔"),
        ("zh_female_xiaoxue_uranus_bigtts", "儿童绘本"), ("zh_female_mengyatou_uranus_bigtts", "萌丫头"),
        ("zh_female_peiqi_uranus_bigtts", "佩奇猪"), ("zh_female_yingtaowanzi_uranus_bigtts", "樱桃丸子"),
        ("zh_male_lubanqihao_uranus_bigtts", "鲁班七号"), ("zh_female_mizai_uranus_bigtts", "黑猫侦探社咪仔"),
    ]),
    ("角色", [
        ("zh_male_sunwukong_uranus_bigtts", "猴哥"), ("zh_male_xionger_uranus_bigtts", "熊二"),
        ("zh_male_zhubajie_uranus_bigtts", "猪八戒"), ("zh_male_tangseng_uranus_bigtts", "唐僧"),
        ("zh_female_popo_uranus_bigtts", "婆婆"), ("zh_female_gaolengyujie_uranus_bigtts", "高冷御姐"),
        ("zh_male_aojiaobazong_uranus_bigtts", "傲娇霸总"), ("zh_female_tvbnv_uranus_bigtts", "TVB 女声"),
        ("zh_male_yizhipiannan_uranus_bigtts", "译制片男"), ("zh_male_xuanyijieshuo_uranus_bigtts", "悬疑解说"),
    ]),
    ("英文", [
        ("en_male_tim_uranus_bigtts", "Tim"), ("en_female_dacey_uranus_bigtts", "Dacey"), ("en_female_stokie_uranus_bigtts", "Stokie"),
    ]),
]
VOLC_RESOURCES = ["seed-tts-2.0", "seed-tts-1.0", "seed-icl-2.0", "seed-icl-1.0"]


def is_volc(voice: str) -> bool:
    """Doubao voice IDs end in _bigtts; cloned voices start with S_ (or saturn_ for 2.0 premium voices)."""
    return voice.endswith("_bigtts") or voice.startswith(("S_", "saturn_"))


def volc_resource(voice: str, override: str = "") -> str:
    if override:
        return override
    if voice.startswith("S_"):
        return "seed-icl-2.0"
    if "_uranus_" in voice or voice.startswith("saturn_"):
        return "seed-tts-2.0"
    return "seed-tts-1.0"


def engine_of(voice: str) -> str:
    if is_volc(voice):
        return "volc"
    if voice in {v for v, _ in QWEN_OMNI}:
        return "omni"
    return "qwen_tts"


def catalog() -> list[dict]:
    groups = [
        {"engine": "omni", "label": "千问 Omni 原生音色（延迟最低）", "voices": [{"id": v, "name": n} for v, n in QWEN_OMNI]},
        {"engine": "qwen_tts", "label": "千问 TTS 音色", "voices": [{"id": v, "name": n} for v, n in QWEN_TTS]},
    ]
    for label, voices in VOLC_GROUPS:
        groups.append({"engine": "volc", "label": f"火山引擎·豆包 2.0 · {label}", "voices": [{"id": v, "name": n} for v, n in voices]})
    return groups


def label_of(voice: str) -> str:
    for group in catalog():
        for v in group["voices"]:
            if v["id"] == voice:
                return v["name"]
    return voice
