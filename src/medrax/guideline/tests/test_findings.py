"""Tests for the findings→query mapping (SCP codes → CN guideline query).

    PYTHONPATH=src python -m unittest medrax.guideline.tests.test_findings -v
"""

from __future__ import annotations

import unittest
from pathlib import Path

from medrax.guideline import (GuidelineRetriever, findings_to_query, parse_findings,
                              load_findings_map, ground_by_findings)

CORPUS = Path(__file__).resolve().parent.parent / "corpus" / "ecg_guidelines_cn.jsonl"

# the classifier's 72-code vocabulary (medrax.tools.classification.CLASS_LABELS)
CLASS_LABELS = ['1AVB','2AVB','3AVB','ABQRS','AFIB','AFLT','ALMI','AMI','ANEUR','ASMI','BIGU','CLBBB',
    'CRBBB','DIG','EL','HVOLT','ILBBB','ILMI','IMI','INJAL','INJAS','INJIL','INJIN','INJLA','INVT','IPLMI',
    'IPMI','IRBBB','ISCAL','ISCAN','ISCAS','ISCIL','ISCIN','ISCLA','ISC_','IVCD','LAFB','LAO/LAE','LMI',
    'LNGQT','LOWT','LPFB','LPR','LVH','LVOLT','NDT','NORM','NST_','NT_','PAC','PACE','PMI','PRC(S)','PSVT',
    'PVC','QWAVE','RAO/RAE','RVH','SARRH','SBRAD','SEHYP','SR','STACH','STD_','STE_','SVARR','SVTAC','TAB_',
    'TRIGU','VCLVH','WPW']

# the real cached_classification string seen on the device
_REAL = "['PACE(0.49)', 'ILMI(0.46)', 'AFIB(0.29)', 'QWAVE(0.15)', 'LMI(0.14)', 'ASMI(0.10)']"

_CORPUS_TOPICS = {"atrial_fibrillation", "bradycardia_conduction", "qt_channelopathy",
                  "ischemia_stemi", "svt", "electrolyte",
                  "ventricular_arrhythmia", "hypertrophy_cardiomyopathy"}


class ParseTests(unittest.TestCase):
    def test_parse_repr_string(self):
        d = dict(parse_findings(_REAL))
        self.assertEqual(d["AFIB"], 0.29)
        self.assertEqual(d["PACE"], 0.49)
        self.assertIn("ILMI", d)

    def test_parse_list_with_and_without_prob(self):
        d = dict(parse_findings(["AFIB(0.9)", "3AVB"]))
        self.assertEqual(d["AFIB"], 0.9)
        self.assertIsNone(d["3AVB"])


class MapIntegrityTests(unittest.TestCase):
    def test_every_classifier_label_is_mapped(self):
        fmap = load_findings_map()
        missing = [c for c in CLASS_LABELS if c not in fmap]
        self.assertEqual(missing, [], f"unmapped classifier codes: {missing}")

    def test_actionable_entries_have_terms_and_valid_topic(self):
        for code, e in load_findings_map().items():
            if e.get("topic"):
                self.assertIn(e["topic"], _CORPUS_TOPICS, f"{code} bad topic {e['topic']}")
                self.assertTrue(e.get("cn"), f"{code} actionable but has no CN terms")

    def test_pacemaker_present_is_not_actionable(self):
        # PACE = pacemaker already present, NOT a pacing indication → no grounding
        self.assertIsNone(load_findings_map()["PACE"]["topic"])


class QueryTests(unittest.TestCase):
    def test_afib_maps_to_cn_and_topic(self):
        q, topics = findings_to_query(["AFIB(0.9)"])
        self.assertIn("房颤", q)
        self.assertEqual(topics, {"atrial_fibrillation"})

    def test_3avb_maps_to_conduction(self):
        q, topics = findings_to_query(["3AVB(0.8)"])
        self.assertIn("三度房室传导阻滞", q)
        self.assertEqual(topics, {"bradycardia_conduction"})

    def test_nonactionable_only_yields_empty(self):
        q, topics = findings_to_query(["PACE(0.9)", "SR(0.8)", "NORM(0.7)", "STACH(0.5)"])
        self.assertEqual(q, "")
        self.assertEqual(topics, set())

    def test_prob_threshold_filters(self):
        # AFIB present but below threshold → dropped
        q, _ = findings_to_query(["AFIB(0.10)"], prob_threshold=0.3)
        self.assertEqual(q, "")
        q2, _ = findings_to_query(["AFIB(0.90)"], prob_threshold=0.3)
        self.assertIn("房颤", q2)

    def test_real_device_string_maps(self):
        # PACE non-actionable; ILMI/AFIB/etc. → ischemia + AF terms
        q, topics = findings_to_query(_REAL)
        self.assertTrue(q)
        self.assertTrue({"atrial_fibrillation", "ischemia_stemi"} & topics)


class EndToEndTests(unittest.TestCase):
    """The whole point: EN labels couldn't retrieve from the CN corpus; the
    mapped CN query can."""

    def setUp(self):
        self.retr = GuidelineRetriever.from_corpus(CORPUS, enable_dense=False)

    def test_raw_labels_retrieve_nothing(self):
        self.assertEqual(self.retr.search("AFIB PACE ILMI", k=3), [])

    def test_mapped_query_retrieves_af_record(self):
        q, _ = findings_to_query(["AFIB(0.9)"])
        topics = {h.record.topic for h in self.retr.search(q, k=5)}
        self.assertIn("atrial_fibrillation", topics, "mapped CN query should retrieve AF-topic guideline")

    def test_mapped_3avb_retrieves_pacing_record(self):
        q, _ = findings_to_query(["3AVB(0.9)"])
        topics = {h.record.topic for h in self.retr.search(q, k=5)}
        self.assertIn("bradycardia_conduction", topics)

    def test_ground_by_findings_scopes_to_topics(self):
        # real device string (AF + ischemia) must NOT surface the electrolyte
        # (hyperkalemia) record despite 心肌/心 bigram bleed — topic scoping fixes it
        hits = ground_by_findings(self.retr, _REAL, k=3)
        self.assertTrue(hits)
        topics = {h.record.topic for h in hits}
        self.assertTrue(topics <= {"atrial_fibrillation", "ischemia_stemi"}, topics)
        self.assertNotIn("hyperkalemia_ecg", [h.record.id for h in hits])

    def test_ground_by_findings_empty_when_nonactionable(self):
        self.assertEqual(ground_by_findings(self.retr, ["PACE(0.9)", "SR(0.8)"]), [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
