"""Offline: DeepSeek drafts the CLINICAL content of a Chinese guideline corpus,
into a fixed structure (slots) we control. We own ids / topics / version-pair
links; DeepSeek fills {text, class, LOE, effective_date, citation} per slot.

Output: a CANDIDATE jsonl (verified=false) + a human-readable summary on stderr.
A human cross-checks Class/LOE + edition years before promoting to the corpus.

    DEEPSEEK_API_KEY=... python src/medrax/guideline/scripts/draft_corpus.py > /tmp/corpus_candidate.jsonl
"""
from __future__ import annotations
import json, os, sys, urllib.request

MODEL, ENDPOINT = "deepseek-v4-pro", "https://api.deepseek.com/chat/completions"

# Structure is human-owned. Each slot: id, topic, section, society, doc_title, version,
# status, supersedes/superseded_by, hint (what recommendation to write).
SLOTS = [
  # ---- 房颤 AF: 中国心房颤动诊断和治疗指南, version pairs 2018↔2023 ----
  {"id":"af_anticoag_2023","topic":"atrial_fibrillation","section":"抗凝治疗","society":"中华医学会心血管病学分会","doc_title":"心房颤动诊断和治疗中国指南","version":"2023","status":"current","supersedes":["af_anticoag_2018"],"superseded_by":[],"hint":"非瓣膜性房颤按 CHA2DS2-VASc 评分抗凝、优先 NOAC 而非华法林的现行推荐"},
  {"id":"af_anticoag_2018","topic":"atrial_fibrillation","section":"抗凝治疗","society":"中华医学会心血管病学分会","doc_title":"心房颤动诊断和治疗中国指南","version":"2018","status":"superseded","supersedes":[],"superseded_by":["af_anticoag_2023"],"hint":"2018 版房颤抗凝推荐（旧版，华法林或 NOAC 并列）"},
  {"id":"af_rate_2023","topic":"atrial_fibrillation","section":"心室率控制","society":"中华医学会心血管病学分会","doc_title":"心房颤动诊断和治疗中国指南","version":"2023","status":"current","supersedes":["af_rate_2018"],"superseded_by":[],"hint":"现行房颤心室率控制目标（宽松 vs 严格）推荐"},
  {"id":"af_rate_2018","topic":"atrial_fibrillation","section":"心室率控制","society":"中华医学会心血管病学分会","doc_title":"心房颤动诊断和治疗中国指南","version":"2018","status":"superseded","supersedes":[],"superseded_by":["af_rate_2023"],"hint":"2018 版房颤心室率控制推荐（旧版）"},
  {"id":"af_ablation_2023","topic":"atrial_fibrillation","section":"节律控制/导管消融","society":"中华医学会心血管病学分会","doc_title":"心房颤动诊断和治疗中国指南","version":"2023","status":"current","supersedes":[],"superseded_by":[],"hint":"症状性阵发性房颤导管消融作为节律控制的现行推荐"},
  # ---- 缓慢性心律失常/传导阻滞 起搏 ----
  {"id":"avb_high_pacing","topic":"bradycardia_conduction","section":"起搏适应证","society":"中华医学会心血管病学分会","doc_title":"心动过缓和传导异常患者的评估与管理中国专家共识","version":"2020","status":"current","supersedes":[],"superseded_by":[],"hint":"三度或高度房室传导阻滞植入永久起搏器的推荐"},
  {"id":"snd_pacing","topic":"bradycardia_conduction","section":"起搏适应证","society":"中华医学会心血管病学分会","doc_title":"心动过缓和传导异常患者的评估与管理中国专家共识","version":"2020","status":"current","supersedes":[],"superseded_by":[],"hint":"症状性窦房结功能障碍（病窦综合征）植入永久起搏器的推荐"},
  {"id":"firstdeg_avb_no_pacing","topic":"bradycardia_conduction","section":"起搏适应证","society":"中华医学会心血管病学分会","doc_title":"心动过缓和传导异常患者的评估与管理中国专家共识","version":"2020","status":"current","supersedes":[],"superseded_by":[],"hint":"无症状一度房室传导阻滞不推荐起搏（Class III）"},
  # ---- 长QT ----
  {"id":"longqt_acquired","topic":"qt_channelopathy","section":"QT延长处理","society":"中华医学会","doc_title":"室性心律失常中国专家共识","version":"2020","status":"current","supersedes":[],"superseded_by":[],"hint":"获得性 QT 延长：停诱因药物、纠正电解质的推荐"},
  {"id":"longqt_congenital_bb","topic":"qt_channelopathy","section":"长QT综合征治疗","society":"中华医学会","doc_title":"室性心律失常中国专家共识","version":"2020","status":"current","supersedes":[],"superseded_by":[],"hint":"先天性长 QT 综合征一线 β 受体阻滞剂的推荐"},
  # ---- STEMI: 中国 STEMI 指南, version pair 2015↔2019 ----
  {"id":"stemi_ppci_2019","topic":"ischemia_stemi","section":"再灌注治疗","society":"中华医学会心血管病学分会","doc_title":"急性ST段抬高型心肌梗死诊断和治疗指南","version":"2019","status":"current","supersedes":["stemi_ppci_2015"],"superseded_by":[],"hint":"发病12h内 STEMI 首选直接 PCI 再灌注、缩短时间的现行推荐"},
  {"id":"stemi_ppci_2015","topic":"ischemia_stemi","section":"再灌注治疗","society":"中华医学会心血管病学分会","doc_title":"急性ST段抬高型心肌梗死诊断和治疗指南","version":"2015","status":"superseded","supersedes":[],"superseded_by":["stemi_ppci_2019"],"hint":"2015 版 STEMI 直接 PCI 再灌注推荐（旧版）"},
  {"id":"stemi_lysis","topic":"ischemia_stemi","section":"再灌注治疗","society":"中华医学会心血管病学分会","doc_title":"急性ST段抬高型心肌梗死诊断和治疗指南","version":"2019","status":"current","supersedes":[],"superseded_by":[],"hint":"无法及时 PCI 时溶栓治疗的推荐"},
  # ---- 室上速 ----
  {"id":"svt_acute_vagal","topic":"svt","section":"急性终止","society":"中华医学会","doc_title":"室上性心动过速相关共识/指南","version":"2021","status":"current","supersedes":[],"superseded_by":[],"hint":"血流动力学稳定的规则窄QRS室上速：迷走刺激、无效用腺苷的推荐"},
  {"id":"svt_ablation","topic":"svt","section":"导管消融","society":"中华医学会","doc_title":"室上性心动过速相关共识/指南","version":"2021","status":"current","supersedes":[],"superseded_by":[],"hint":"反复发作室上速导管消融的推荐"},
  # ---- 电解质 ----
  {"id":"hyperkalemia_ecg","topic":"electrolyte","section":"高钾血症","society":"中华医学会","doc_title":"电解质紊乱心电图与处理相关共识","version":"2020","status":"current","supersedes":[],"superseded_by":[],"hint":"高钾血症心电图表现 + 出现改变时静脉钙剂稳定心肌膜的紧急处理推荐"},
  {"id":"hypokalemia_ecg","topic":"electrolyte","section":"低钾血症","society":"中华医学会","doc_title":"电解质紊乱心电图与处理相关共识","version":"2020","status":"current","supersedes":[],"superseded_by":[],"hint":"低钾血症心电图表现（U波等）+ 补钾处理推荐"},
]


def call(prompt, max_tokens=12000):
    body = json.dumps({"model": MODEL, "messages": [{"role": "user", "content": prompt}],
                       "max_tokens": max_tokens, "temperature": 0}).encode()
    req = urllib.request.Request(ENDPOINT, data=body, headers={
        "Authorization": f"Bearer {os.environ['DEEPSEEK_API_KEY']}", "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=240) as r:
        return json.loads(r.read())


def main():
    ask = [{"id": s["id"], "条件": s["hint"], "指南": s["doc_title"], "版本": s["version"], "章节": s["section"]} for s in SLOTS]
    prompt = (
        "你是中国心血管临床指南专家。下面每个条目要写成一条**中文指南推荐**。请严格依据所列中国指南/共识的对应版本，"
        "为每个 id 填写：\n"
        "- text: 1-2 句中文推荐原意（凝练、可作为临床建议）\n"
        "- class: 推荐类别，取值 I / IIa / IIb / III\n"
        "- LOE: 证据级别，取值 A / B / C\n"
        "- effective_date: 该版本年份(YYYY)\n"
        "- citation: 真实指南名称+版本年份（如『中华医学会心血管病学分会. 心房颤动诊断和治疗中国指南(2023)』）\n"
        "- note: 你不确定的地方（尤其 class/LOE 或版本年份吃不准就写明）\n"
        "严格只返回 JSON 对象，键为 id，值为上述字段对象；不要 markdown、不要多余文字。\n\n"
        + json.dumps(ask, ensure_ascii=False, indent=1))
    resp = call(prompt)
    ch = resp["choices"][0]
    print(f"[finish={ch.get('finish_reason')} completion_tok={resp.get('usage',{}).get('completion_tokens')}]", file=sys.stderr)
    content = ch["message"]["content"].strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    filled = json.loads(content)

    print("id | ver | status | class/LOE | text | note", file=sys.stderr)
    for s in SLOTS:
        f = filled.get(s["id"], {})
        rec = {"id": s["id"], "society": s["society"], "doc_title": s["doc_title"],
               "version": s["version"], "effective_date": str(f.get("effective_date", s["version"])),
               "status": s["status"], "topic": s["topic"], "section": s["section"],
               "class": f.get("class", "?"), "LOE": f.get("LOE", "?"), "lang": "zh",
               "citation": f.get("citation", ""), "text": f.get("text", ""),
               "supersedes": s["supersedes"], "superseded_by": s["superseded_by"], "verified": False}
        print(json.dumps(rec, ensure_ascii=False))  # candidate jsonl -> stdout
        print(f"  {s['id']:24s} {s['version']} {s['status']:10s} {f.get('class','?')}/{f.get('LOE','?')} | "
              f"{f.get('text','')[:34]}… | note={f.get('note','')}", file=sys.stderr)


if __name__ == "__main__":
    main()
