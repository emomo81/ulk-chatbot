"""
ULK Fees Assistant - retrieval engine.

Answers questions using ONLY the data in knowledge_base.json.
Works fully offline (no API key needed). Keeps a small conversation
memory so follow-ups like "and in Gisenyi?" or "evening?" work.
"""
from __future__ import annotations

import difflib
import json
import math
import random
import re
from dataclasses import dataclass, field
from pathlib import Path

KB_PATH = Path(__file__).parent / "knowledge_base.json"

STOPWORDS = {
    "a", "an", "the", "of", "in", "and", "or", "for", "to", "is", "are", "was", "be", "what", "whats",
    "how", "much", "many", "fee", "tuition", "cost", "costs", "price", "charge", "amount", "do", "does",
    "i", "me", "my", "we", "you", "your", "please", "tell", "about", "at", "on", "per", "ulk", "university",
    "independent", "can", "could", "would", "will", "want", "need", "know", "give", "show", "it", "this",
    "that", "with", "there", "pay", "paid", "rwf", "frw", "francs", "student", "studying", "study",
    "programme", "program", "course", "degree", "bachelor", "undergraduate", "year", "term", "total",
    "kigali", "gisenyi", "rubavu", "day", "evening", "weekend", "holiday", "night", "campus",
    "also", "same", "if", "so", "get", "like", "which", "who", "when", "where", "why", "much", "us",
    "should", "have", "has", "am", "hello", "hi", "thanks", "thank", "am", "its", "by", "from", "as",
    "sir", "madam", "dear", "full", "fees", "school", "department", "faculty", "list", "all", "offer",
}

LEVEL_WORDS = {
    "phd": "phd", "doctorate": "phd", "doctoral": "phd",
    "master": "masters", "msc": "masters", "mba": "masters", "postgraduate": "masters", "llm": "masters",
    "btech": "upi_btech",
    "diploma": "upi_advanced_diploma", "a1": "upi_advanced_diploma",
    "bachelor": "undergraduate", "undergraduate": "undergraduate", "bba": "undergraduate",
}

PAYMENT_WORDS = {"bank", "account", "pay", "payment", "deposit", "equity", "unguka", "bk", "transfer money"}
CONTACT_WORDS = {"contact", "phone", "call", "email", "mail", "website", "address", "reach", "telephone",
                 "whatsapp", "office", "box"}
LIST_WORDS = {"program", "programme", "course", "offer", "list", "available", "option", "school",
              "faculty", "department"}
GRAD_HINT = {"supervision", "dissertation", "gown", "graduation", "thesis", "project"}


def normalize(text: str) -> str:
    t = text.lower()
    t = t.replace("&", " and ").replace("’", "'")
    t = re.sub(r"\bb[\s\.-]?tech\b", "btech", t)
    t = re.sub(r"bachelor'?s? of technology", "btech", t)
    t = re.sub(r"\bph\.?\s?d\b", "phd", t)
    t = re.sub(r"\bi\s?and\s?m\b", "imbank", t)
    t = re.sub(r"master'?s", "master", t)
    t = re.sub(r"bachelor'?s", "bachelor", t)
    t = re.sub(r"[^a-z0-9\s]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def stem(tok: str) -> str:
    if len(tok) > 4 and tok.endswith("ies"):
        return tok[:-3] + "y"
    if len(tok) > 4 and tok.endswith("s") and not tok.endswith("ss"):
        return tok[:-1]
    return tok


def tokens(text: str) -> list[str]:
    return [stem(w) for w in normalize(text).split()]


def money(n: int | float) -> str:
    return f"{int(n):,} Rwf"


@dataclass
class Memory:
    """Conversation memory kept per user session."""
    last_programs: list[str] = field(default_factory=list)
    last_campus: str | None = None
    last_schedule: str | None = None
    last_topic: str | None = None
    turns: int = 0


class FeesBot:
    def __init__(self, kb_path: Path | str = KB_PATH):
        with open(kb_path, encoding="utf-8") as f:
            self.kb = json.load(f)
        self.programs = {p["id"]: p for p in self.kb["programs"]}
        self.items = {i["id"]: i for i in self.kb["non_tuition_fees"]}
        self.levels = self.kb["levels"]
        # pseudo-item for graduation fees so it competes with other items
        g = self.kb["graduation_fees"]
        self.items["__graduation__"] = {"id": "__graduation__", "name": g["name"], "aliases": g["aliases"]}
        self._full_hits: set = set()
        self._build_index()

    # ------------------------------------------------------------------ index
    def _build_index(self):
        def alias_sets(entries):
            out = {}
            for e in entries.values():
                sets = []
                for a in e["aliases"] + [e["name"]]:
                    ts = [t for t in tokens(a) if t not in STOPWORDS or t in LEVEL_WORDS]
                    if ts:
                        sets.append(frozenset(ts))
                out[e["id"]] = sets
            return out

        self.prog_alias = alias_sets(self.programs)
        self.item_alias = alias_sets(self.items)
        self.school_alias = {s["id"]: [frozenset(tokens(a)) - {"school", "faculty"} or frozenset(tokens(a))
                                       for a in s["aliases"]] for s in self.kb["schools"]}

        def idf(alias_map):
            df = {}
            for sets in alias_map.values():
                for t in set().union(*sets) if sets else set():
                    df[t] = df.get(t, 0) + 1
            n = len(alias_map)
            return {t: math.log(n / c) + 1 for t, c in df.items()}

        self.prog_idf = idf(self.prog_alias)
        self.item_idf = idf(self.item_alias)

        vocab = set(self.prog_idf) | set(self.item_idf) | set(LEVEL_WORDS) | PAYMENT_WORDS | CONTACT_WORDS
        vocab |= {"kigali", "gisenyi", "rubavu", "evening", "weekend", "holiday", "campus", "international",
                  "graduation", "supervision", "program", "programme", "course", "account", "accounting"}
        for intent in self.kb["intents"] + self.kb["faqs"]:
            for p in intent["patterns"]:
                vocab |= set(tokens(p))
        self.vocab = sorted(vocab)

    def _correct(self, toks: list[str]) -> list[str]:
        out = []
        for t in toks:
            if len(t) >= 4 and t not in self.vocab and not t.isdigit():
                m = difflib.get_close_matches(t, self.vocab, n=1, cutoff=0.82)
                out.append(m[0] if m else t)
            else:
                out.append(t)
        return out

    def _score(self, qset: set, alias_map: dict, idf: dict) -> dict[str, float]:
        scores = {}
        self._full_hits = set()
        for eid, sets in alias_map.items():
            best = 0.0
            for s in sets:
                matched = s & qset
                if not matched:
                    continue
                mw = sum(idf.get(t, 1) for t in matched)
                tw = sum(idf.get(t, 1) for t in s)
                cov = mw / tw
                sc = mw if cov >= 0.999 else mw * cov * 0.7
                if cov >= 0.999:
                    self._full_hits.add(eid)
                best = max(best, sc)
            if best > 0:
                scores[eid] = best
        return scores

    # --------------------------------------------------------------- entities
    @staticmethod
    def _detect_campus(norm: str) -> str | None:
        n = norm.replace("bank of kigali", " ").replace("kigali independent", " ")
        if re.search(r"\b(gisenyi|rubavu)\b", n):
            return "Gisenyi"
        if re.search(r"\bkigali\b", n):
            return "Kigali"
        return None

    @staticmethod
    def _detect_schedule(norm: str) -> str | None:
        if re.search(r"\bweekends?\b", norm):
            return "Weekend"
        if re.search(r"\b(evening|night)\b", norm) and not re.fullmatch(r"good evening", norm):
            return "Evening"
        if re.search(r"\bholidays?\b", norm):
            return "Holidays"
        if re.search(r"\b(day|daytime|morning)\b", norm) and "per day" not in norm:
            return "Day"
        return None

    @staticmethod
    def _detect_level(toks: list[str]) -> str | None:
        found = [LEVEL_WORDS[t] for t in toks if t in LEVEL_WORDS]
        if "upi" in toks or "polytechnic" in toks:
            if "btech" in found:
                return "upi_btech"
            if "upi_advanced_diploma" in found:
                return "upi_advanced_diploma"
            return "upi"
        return found[0] if found else None

    @staticmethod
    def _has_phrase(norm: str, patterns: list[str]) -> bool:
        padded = f" {norm} "
        for p in patterns:
            pn = " ".join(tokens(p))
            if f" {pn} " in f" {' '.join(stem(w) for w in padded.split())} ":
                return True
        return False

    # ---------------------------------------------------------------- answer
    def answer(self, message: str, mem: Memory | None = None) -> tuple[str, dict]:
        """Return (markdown_answer, debug_info). Updates memory in place."""
        mem = mem or Memory()
        mem.turns += 1
        norm = normalize(message)
        toks = self._correct([stem(w) for w in norm.split()])
        norm_c = " ".join(toks)
        tset = set(toks)
        content = {t for t in toks if t not in STOPWORDS}

        campus = self._detect_campus(norm)
        schedule = self._detect_schedule(re.sub(r"\bgood (morning|afternoon|evening)\b", " ", norm).strip())
        level = self._detect_level(toks)

        info = {"campus": campus, "schedule": schedule, "level": level}
        if not norm:
            return self._fallback(), info

        # ---- small talk (only when message is basically just that)
        fillers = {"ulk", "there", "so", "much", "very", "a", "lot", "you", "ok", "okay", "and", "great",
                   "bot", "assistant", "dear", "sir", "madam", "all", "everyone"}
        stripped = " ".join(t for t in toks if t not in fillers)
        for intent in self.kb["intents"]:
            pats = intent["patterns"]
            pset = {" ".join(tokens(p)) for p in pats}
            singles = {p for p in pset if " " not in p}
            if intent["tag"] == "about":
                if self._has_phrase(norm_c, pats) and len(toks) <= 6:
                    info["intent"] = "about"
                    return random.choice(intent["responses"]), info
            elif stripped and (stripped in pset or all(t in singles for t in stripped.split())):
                info["intent"] = intent["tag"]
                return random.choice(intent["responses"]), info

        # ---- topical detection
        wants_payment = bool(tset & {"bank", "deposit", "equity", "unguka", "bk", "imbank"}) or \
            "account number" in norm or bool(re.search(r"\b(how|where) (do i |can i |to )?pay\b", norm)) or \
            "payment" in tset
        wants_contact = bool(tset & CONTACT_WORDS) or "phone number" in norm
        wants_list = bool(tset & LIST_WORDS) and not (content - LIST_WORDS - set(LEVEL_WORDS) - {"upi", "polytechnic"})

        self._full_hits = set()
        specific = content - set(LEVEL_WORDS) - {"upi", "polytechnic"}
        prog_scores = self._score(content | (tset & set(LEVEL_WORDS)), self.prog_alias, self.prog_idf) if specific else {}
        school = self._match_school(tset)
        if school and not self._full_hits:
            prog_scores = {}
        prog_full = set(self._full_hits) & set(prog_scores)
        item_scores = self._score(content, self.item_alias, self.item_idf)

        # level filter for programs
        if level and prog_scores:
            lv = (lambda p: p["level"].startswith("upi")) if level == "upi" else (lambda p: p["level"] == level)
            filtered = {k: v for k, v in prog_scores.items() if lv(self.programs[k])}
            if filtered:
                prog_scores = filtered

        parts: list[str] = []

        # payment
        if wants_payment:
            parts.append(self._bank_answer(campus))
            mem.last_topic = "payment"

        # contact
        if wants_contact and not wants_payment:
            parts.append(self._contact_answer())
            mem.last_topic = "contact"

        # FAQs
        for faq in self.kb["faqs"]:
            if self._has_phrase(norm_c, faq["patterns"]):
                if faq["tag"] == "campuses" and (prog_scores or wants_payment):
                    continue
                if faq["tag"] == "payment_frequency" and prog_scores:
                    continue
                parts.append(faq["answer"])
                mem.last_topic = faq["tag"]
                # words used by the FAQ should not also trigger programme / fee look-ups
                faq_toks = set().union(*[set(tokens(p)) for p in faq["patterns"]])
                rest = content - faq_toks - {"change", "national"}
                prog_scores = self._score(rest, self.prog_alias, self.prog_idf) if rest - set(LEVEL_WORDS) else {}
                item_scores = self._score(rest, self.item_alias, self.item_idf) if rest else {}
                break

        # non-tuition items
        best_item = max(item_scores.values()) if item_scores else 0
        best_prog = max(prog_scores.values()) if prog_scores else 0
        show_item = best_item >= 2.5 and best_item >= best_prog * 0.6
        if show_item:
            top = [k for k, v in sorted(item_scores.items(), key=lambda x: -x[1]) if v >= best_item * 0.85][:4]
            parts.append(self._items_answer(top, level))
            mem.last_topic = "item"
            info["items"] = top

        # programmes (also shown alongside a fee item when the programme is named explicitly)
        if (not show_item or prog_full) and best_prog > 0 and not (wants_payment and best_prog < 4):
            ranked = sorted(prog_scores.items(), key=lambda x: -x[1])
            primary = [k for k, v in ranked if v >= best_prog * 0.85]
            related = [k for k, v in ranked if best_prog * 0.4 <= v < best_prog * 0.85][:6]
            # school-wide question e.g. "school of law fees" -> fine, multiple primaries
            parts.append(self._programs_answer(primary, campus, schedule, related))
            mem.last_programs = primary
            mem.last_campus, mem.last_schedule = campus, schedule
            mem.last_topic = "program"
            info["programs"] = primary

        # school / level listing
        if not parts:
            if school:
                parts.append(self._school_answer(school, campus, schedule))
                mem.last_topic = "school"
            elif level and mem.last_programs and self._is_followup(norm) and not wants_list and \
                    (ids := self._same_subject_other_level(mem.last_programs, level)):
                # ---- MEMORY: same subject, different level ("finance" -> "what about master's?")
                parts.append("_Continuing with your previous question…_\n\n" +
                             self._programs_answer(ids, campus, schedule, []))
                mem.last_programs = ids
                info["used_memory"] = True
            elif level and (wants_list or len(content - set(LEVEL_WORDS)) == 0 or "fee" in tset or "much" in tset):
                parts.append(self._level_answer(level, campus, schedule))
                mem.last_topic = "level"
            elif wants_list:
                parts.append(self._all_programs_answer())
                mem.last_topic = "list"
            elif tset & GRAD_HINT:
                parts.append(self._graduation_answer(level))
            elif mem.last_programs and (campus or schedule or self._is_followup(norm)):
                # ---- MEMORY: follow-up on previous programme(s)
                c = campus or mem.last_campus
                sch = schedule or mem.last_schedule
                parts.append("_Continuing with your previous question…_\n\n" +
                             self._programs_answer(mem.last_programs, c, sch, []))
                info["used_memory"] = True

        if campus:
            mem.last_campus = campus
        if schedule:
            mem.last_schedule = schedule

        if not parts:
            return self._fallback(), info
        return "\n\n---\n\n".join(parts), info

    @staticmethod
    def _is_followup(norm: str) -> bool:
        return bool(re.match(r"^(and|what about|how about|ok and|also|then|for)\b", norm)) or \
            bool(re.search(r"\b(it|that|this one|same|them|those)\b", norm)) or len(norm.split()) <= 3

    def _same_subject_other_level(self, last_ids: list[str], level: str) -> list[str]:
        generic = set(LEVEL_WORDS) | STOPWORDS | {"business", "administration", "science", "education"}
        subj = set()
        for i in last_ids:
            subj |= {t for t in tokens(self.programs[i]["name"]) if t not in generic}
        if not subj:
            return []
        scores = self._score(subj, self.prog_alias, self.prog_idf)
        ok = (lambda p: p["level"].startswith("upi")) if level == "upi" else (lambda p: p["level"] == level)
        scores = {k: v for k, v in scores.items() if ok(self.programs[k]) and k not in last_ids}
        if not scores:
            return []
        best = max(scores.values())
        return [k for k, v in sorted(scores.items(), key=lambda x: -x[1]) if v >= best * 0.5][:4]

    def _match_school(self, tset: set):
        best, best_s = None, 0
        for s in self.kb["schools"]:
            for a in self.school_alias[s["id"]]:
                if a and a <= tset and len(a) > best_s:
                    best, best_s = s, len(a)
        return best

    # ---------------------------------------------------------- formatters
    def _rate_rows(self, p, campus, schedule):
        t = p["tuition"]
        rows = t.get("rates", [])
        out = rows
        if campus:
            out = [r for r in out if r["campus"] in (campus, "All")]
        if schedule:
            s = [r for r in out if schedule.lower().rstrip("s") in r["schedule"].lower()]
            out = s or out
        return out or rows

    def _program_block(self, p, campus, schedule) -> str:
        t = p["tuition"]
        lvl = self.levels.get(p["level"], p["level"])
        head = f"### {p['name']}\n**{lvl}** · {p['school']}" + (f" · {p['department']}" if p.get("department") else "")
        head += f"\n\n**Duration:** {p['duration']}"
        lines = [head]
        years = p.get("duration_years", [1])

        if t["unit"] == "term":
            rows = self._rate_rows(p, campus, schedule)
            tbl = ["| Campus | Schedule | Per term | Per year | Full programme (est.) |",
                   "|---|---|---|---|---|"]
            for r in rows:
                full = " / ".join(f"{money(r['per_year'] * y)} ({y} yrs)" for y in years)
                tbl.append(f"| {r['campus']} | {r['schedule']} | **{money(r['per_term'])}** | {money(r['per_year'])} | {full} |")
            lines.append("\n".join(tbl))
        elif t["unit"] == "year":
            rows = self._rate_rows(p, None, schedule)
            tbl = ["| Programme | Per year | Full programme (est.) |", "|---|---|---|"]
            for r in rows:
                y = years[0]
                tbl.append(f"| {r['schedule']} | **{money(r['per_year'])}** | {money(r['per_year'] * y)} ({y} yr{'s' if y > 1 else ''}) |")
            lines.append("\n".join(tbl))
        elif t["unit"] == "module":
            lines.append(f"**{money(t['per_module'])} per module** × {t['modules']} modules = "
                         f"**{money(t['estimated_total'])}** estimated total tuition")
        elif t["unit"] == "semester":
            lines.append(f"**{money(t['per_semester'])} per semester** · {money(t['per_year'])} per year · "
                         f"≈ {money(t['estimated_total'])} for 4 years\n\n"
                         f"Application fee: **{money(p['application_fee'])}** · Admission fee: **{money(p['admission_fee'])}**")

        grad = self.kb["graduation_fees"]["by_level"].get(p["level"])
        extras = []
        if grad:
            extras.append(f"Supervision, Degree & Gown fees: **{money(grad)}** (not refundable)")
        if p["level"] != "phd":
            extras.append(f"Application: {money(self.items['application_fee']['amount'])} · "
                          f"Registration: {money(self.items['registration_fee']['amount'])}")
        if extras:
            lines.append("- " + "  \n- ".join(extras))
        if p.get("notes") and campus in (None, "Gisenyi") and schedule != "Day":
            lines.append(f"Note: _{p['notes']}_")
        return "\n\n".join(lines)

    def _programs_answer(self, ids, campus, schedule, related) -> str:
        progs = [self.programs[i] for i in ids]
        # Many programmes share identical fees (e.g. "law" -> Public & Private law). Merge them.
        groups: dict[str, list] = {}
        for p in progs:
            key = json.dumps([p["level"], p["school"], p["duration"], p["tuition"]], sort_keys=True)
            groups.setdefault(key, []).append(p)
        blocks = []
        for plist in list(groups.values())[:5]:
            if len(plist) == 1:
                blocks.append(self._program_block(plist[0], campus, schedule))
            else:
                merged = dict(plist[0])
                merged["name"] = " · ".join(p["name"] for p in plist)
                merged["department"] = None
                blocks.append(self._program_block(merged, campus, schedule) +
                              "\n\n_These programmes share the same fees._")
        out = "\n\n".join(blocks)
        filt = []
        if campus:
            filt.append(f"campus **{campus}**")
        if schedule:
            filt.append(f"schedule **{schedule}**")
        if filt:
            out = f"Showing results for {' and '.join(filt)}:\n\n" + out
        if any(p["level"] == "undergraduate" for p in progs):
            out += "\n\nNote: _Fees apply to Year One students for 2026–2027. Ongoing students (Year 2–4): no change. Same fees for national & international students._"
        if related:
            names = ", ".join(f"*{self.programs[r]['name']}*" for r in related)
            out += f"\n\n**Related programmes:** {names} — just ask!"
        return out

    def _items_answer(self, ids, level) -> str:
        rows = ["| Fee | Amount | Notes |", "|---|---|---|"]
        extra = []
        for i in ids:
            if i == "__graduation__":
                extra.append(self._graduation_answer(level))
                continue
            it = self.items[i]
            rows.append(f"| {it['name']} | **{money(it['amount'])}** | {it['comment'] or '—'} |")
            if i == "application_fee":
                extra.append("_PhD applicants: Application fee **100,000 Frw** and Admission fee **100,000 Frw**._")
        out = "### Other (non-tuition) fees\n\n" + "\n".join(rows) if len(rows) > 2 else ""
        if extra:
            out = (out + "\n\n" if out else "") + "\n\n".join(extra)
        return out

    def _graduation_answer(self, level) -> str:
        g = self.kb["graduation_fees"]
        rows = ["| Level | Amount |", "|---|---|"]
        for lv, amt in g["by_level"].items():
            if level and level not in (lv, "upi") and not (level == "upi" and lv.startswith("upi")):
                continue
            rows.append(f"| {self.levels[lv]} | **{money(amt)}** |")
        if len(rows) == 2:
            rows += [f"| {self.levels[lv]} | **{money(a)}** |" for lv, a in g["by_level"].items()]
        return f"### {g['name']}\n\n" + "\n".join(rows) + "\n\n_Not refundable._"

    def _bank_answer(self, campus) -> str:
        accts = self.kb["bank_accounts"]
        out = ["### ULK bank accounts for fee payment"]
        for c, lst in accts.items():
            if campus and c != campus:
                continue
            rows = ["| Bank | Account number |", "|---|---|"] + [f"| {a['bank']} | `{a['account']}` |" for a in lst]
            out.append(f"**{c} Campus**\n\n" + "\n".join(rows))
        out.append("_Keep your bank slip. A proof of payment document from ULK costs 5,000 Rwf._")
        return "\n\n".join(out)

    def _contact_answer(self) -> str:
        s = self.kb["school"]
        phones = "  \n".join(s["phones"])
        return (f"### Contact {s['name']}\n\n{phones}\n\n**Email:** {s['email']}  \n"
                f"**Website:** {s['website']}  \n**Address:** {s['address']}, {s['country']}")

    def _school_answer(self, school, campus, schedule) -> str:
        if school["id"] == "upi":
            return self._level_answer("upi", campus, schedule)
        ids = [p["id"] for p in self.kb["programs"] if p["school"] == school["name"]]
        return f"## {school['name']}\n\n" + self._programs_answer(ids, campus, schedule, [])

    def _level_answer(self, level, campus, schedule) -> str:
        if level == "undergraduate":
            rows = ["| School | Programmes | Kigali Day | Kigali Evening | Gisenyi Day | Gisenyi Evening |",
                    "|---|---|---|---|---|---|"]
            seen = {}
            for p in self.kb["programs"]:
                if p["level"] != "undergraduate":
                    continue
                key = (p["school"], p.get("department", ""), json.dumps(p["tuition"]))
                seen.setdefault(key, []).append(p)
            for (school, dep, _), plist in seen.items():
                r = plist[0]["tuition"]["rates"]
                cell = lambda c, ev: next((f"{money(x['per_term'])}/term" for x in r if x["campus"] == c and
                                           (("Day" == x["schedule"]) if not ev else ("Evening" in x["schedule"] or "Holidays" in x["schedule"]))), "—")
                names = ", ".join(p["name"].replace("Bachelor of Education in ", "") for p in plist)
                if "Holidays" in r[0]["schedule"]:
                    rows.append(f"| {school} | {names} | {money(r[0]['per_term'])}/term (Day & Holidays) | — | {money(r[1]['per_term'])}/term (Day & Holidays) | — |")
                else:
                    rows.append(f"| {school} | {names} | {cell('Kigali', False)} | {cell('Kigali', True)} | {cell('Gisenyi', False)} | {cell('Gisenyi', True)} |")
            return ("### Undergraduate tuition 2026–2027 (Year One)\n\n" + "\n".join(rows) +
                    "\n\n_3 terms per year. Supervision, Degree & Gown fees: 250,000 Rwf. Ask about a specific programme for full details._")
        if level in ("upi", "upi_advanced_diploma", "upi_btech"):
            ids = [p["id"] for p in self.kb["programs"]
                   if (level == "upi" and p["level"].startswith("upi")) or p["level"] == level]
            return "## ULK Polytechnic Institute (UPI)\n\n" + self._programs_answer(ids, campus, schedule, [])
        if level == "masters":
            rows = ["| Programme | Modules | Per module | Est. total |", "|---|---|---|---|"]
            for p in self.kb["programs"]:
                if p["level"] == "masters":
                    t = p["tuition"]
                    rows.append(f"| {p['name']} | {t['modules']} | {money(t['per_module'])} | {money(t['estimated_total'])} |")
            return ("### Master's programmes (2 years)\n\n" + "\n".join(rows) +
                    "\n\nSupervision (Project), Degree & Gown fees: **360,000 Rwf** (not refundable)")
        if level == "phd":
            ids = [p["id"] for p in self.kb["programs"] if p["level"] == "phd"]
            return self._programs_answer(ids, campus, schedule, [])
        return self._all_programs_answer()

    def _all_programs_answer(self) -> str:
        out = ["### Programmes in the 2026–2027 fees structure"]
        by: dict[str, dict[str, list]] = {}
        for p in self.kb["programs"]:
            by.setdefault(self.levels[p["level"]], {}).setdefault(p["school"], []).append(p["name"])
        for lvl, schools in by.items():
            out.append(f"**{lvl}**")
            for s, names in schools.items():
                out.append(f"- _{s}_: " + ", ".join(names))
        out.append("Ask me about any of them, e.g. *\"Networking fees in Gisenyi\"*.")
        return "\n\n".join(out)

    def _fallback(self) -> str:
        return random.choice(self.kb["fallback"]["responses"])

    # ----------------------------------------------------- for LLM grounding
    def compact_kb(self) -> str:
        """KB without search aliases, for use as LLM context."""
        kb = json.loads(json.dumps(self.kb))
        for p in kb["programs"]:
            p.pop("aliases", None)
        for i in kb["non_tuition_fees"]:
            i.pop("aliases", None)
        for s in kb["schools"]:
            s.pop("aliases", None)
        kb["graduation_fees"].pop("aliases", None)
        kb.pop("intents", None)
        return json.dumps(kb, ensure_ascii=False, separators=(",", ":"))
