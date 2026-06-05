"""Stage 1: DeepSeek generates an expanded Chinese guideline corpus (~150 recs)
per a human-owned TOPIC_PLAN. We own topics + id assignment + version-pair
linking (via pair_key, not cross-ids — DeepSeek can't break links). DeepSeek
fills clinical content + a generation-time confidence/note (a light self-flag).

Stage 2 (deepseek self-check) + Stage 3 (Claude reviews conflicts) are separate.
Everything stays verified=false until a clinician passes it.

Output: /tmp/corpus_v2_candidate.jsonl  (one record per line)
    DEEPSEEK_API_KEY=... python src/medrax/guideline/scripts/draft_corpus_v2.py
"""
from __future__ import annotations
import json, os, sys, time, urllib.request

MODEL, ENDPOINT = "deepseek-v4-pro", "https://api.deepseek.com/chat/completions"
OUT = "/tmp/corpus_v2_candidate.jsonl"

# topic -> (n_target, instruction). version pairs: tell DeepSeek to emit BOTH the
# current and the superseded record with the SAME pair_key + status + version.
TOPIC_PLAN = {
 "atrial_fibrillation": (18,
  "中国房颤指南(现行 2023《心房颤动诊断和治疗中国指南》; 旧版 2018《心房颤动:目前的认识和治疗的建议》)。覆盖:抗凝(CHA2DS2-VASc/NOAC vs 华法林/出血与拮抗)、心室率控制、节律控制与导管消融、电复律及复律前抗凝、房扑。"
  "其中【抗凝】和【心室率控制】各做一组版本对(2023 current + 2018 superseded, 同 pair_key)。"),
 "bradycardia_conduction": (16,
  "中国《心动过缓和传导异常患者的评估与管理中国专家共识(2020)》。覆盖:一/二/三度房室传导阻滞起搏适应证、病态窦房结综合征起搏、束支/分支阻滞、双束支阻滞、无症状一度AVB不起搏(III类)、CRT 适应证。"),
 "ventricular_arrhythmia": (16,
  "中国《室性心律失常中国专家共识(2020)》及相关。覆盖:室性早搏处理与消融、非持续性室速、持续性单形/多形室速急性处理、ICD 一级与二级预防适应证、特发性室速导管消融、电风暴。"),
 "qt_channelopathy": (12,
  "中国室性心律失常/遗传性心律失常共识。覆盖:获得性QT延长(停诱因药+纠正电解质)、先天性长QT综合征(β阻滞剂一线/ICD)、Brugada综合征、CPVT、尖端扭转型室速急性处理。"),
 "ischemia_stemi": (20,
  "中国 STEMI 指南(现行 2019、旧版 2015《急性ST段抬高型心肌梗死诊断和治疗指南》)与 NSTE-ACS 指南。覆盖:STEMI 直接PCI再灌注、溶栓、转运PCI、抗血小板双联、NSTE-ACS 危险分层与介入时机、心梗后二级预防。"
  "其中【STEMI 直接PCI再灌注】做一组版本对(2019 current + 2015 superseded, 同 pair_key)。"),
 "svt": (12,
  "中国《室上性心动过速诊断及治疗中国专家共识(2021)》。覆盖:窄QRS室上速急性终止(迷走/腺苷)、血流动力学不稳定电复律、AVNRT/AVRT 导管消融、预激综合征(WPW)处理与消融、长期药物。"),
 "electrolyte": (12,
  "中国电解质紊乱心电图与处理相关共识。覆盖:高钾血症(心电图改变+静脉钙剂/降钾)、低钾血症(补钾)、高钙/低钙血症心电图、低镁血症、洋地黄中毒。"),
 "hypertrophy_cardiomyopathy": (14,
  "中国《肥厚型心肌病诊断与治疗指南》及左室肥厚相关。覆盖:左室肥厚评估、肥厚型心肌病(HCM)诊断、HCM 猝死风险评估与 ICD 一级预防、流出道梗阻药物/室间隔减容治疗、运动建议。"),
}


def call(prompt, max_tokens=16000):
    body = json.dumps({"model": MODEL, "messages": [{"role": "user", "content": prompt}],
                       "max_tokens": max_tokens, "temperature": 0.2}).encode()
    req = urllib.request.Request(ENDPOINT, data=body, headers={
        "Authorization": f"Bearer {os.environ['DEEPSEEK_API_KEY']}", "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.loads(r.read())


SCHEMA = ('每条记录字段(严格 JSON 数组，元素为对象):'
 '{"section":"章节中文","doc_title":"指南全名","version":"年份","status":"current|superseded",'
 '"pair_key":"版本对的共享键(非版本对填 null)","class":"I|IIa|IIb|III","LOE":"A|B|C",'
 '"citation":"学会+指南名+年份","text":"1-2句中文推荐","confidence":"high|med|low","note":"不确定点(尤其class/LOE/版本)"}')


def main():
    open(OUT, "w").close()
    total = 0
    mult = float(os.environ.get("CORPUS_MULT", "1"))
    for topic, (n0, instr) in TOPIC_PLAN.items():
        n = max(n0, round(n0 * mult))
        prompt = (f"你是中国心血管临床指南专家。请基于下述指南，为主题【{topic}】生成约 {n} 条**中文指南推荐**。\n"
                  f"指南与覆盖范围：{instr}\n"
                  f"{SCHEMA}\n"
                  "要求：依据真实中国指南/共识；class/LOE 按指南；版本对的两条用相同 pair_key、一条 current 一条 superseded；"
                  "不确定就把 confidence 标 low 并在 note 说明。严格只返回 JSON 数组，不要 markdown、不要多余文字。")
        try:
            resp = call(prompt)
            ch = resp["choices"][0]
            content = ch["message"]["content"].strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
            recs = json.loads(content)
        except Exception as e:
            print(f"[{topic}] FAILED: {e}", file=sys.stderr); continue
        with open(OUT, "a", encoding="utf-8") as f:
            for j, r in enumerate(recs, 1):
                r["topic"] = topic
                r["_idx"] = f"{topic}_{j}"
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        total += len(recs)
        print(f"[{topic}] {len(recs)} recs (finish={ch.get('finish_reason')})", file=sys.stderr)
        time.sleep(1)
    print(f"=== GEN DONE: {total} candidate records -> {OUT} ===", file=sys.stderr)


if __name__ == "__main__":
    main()
