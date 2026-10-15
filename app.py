"""One-Page POV Agent: Plan (code) -> Research (parallel) -> Analysis (parallel) -> Writer -> Fact-check (code).
Type a topic, get an evidence ledger (quotes + links + insights) and a one-page POV built only from it.
Run:  pip install streamlit openai requests ddgs
      export LLM_API_KEY=...   (recommended: TAVILY_API_KEY)
      streamlit run app.py
"""
import csv, io, os, re, time
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlparse
import requests, streamlit as st
from openai import (OpenAI, AuthenticationError, NotFoundError, RateLimitError,
                    APIConnectionError, APITimeoutError, InternalServerError)

NL = chr(10)
Q = chr(34)
HEADERS = {"User-Agent": "AgenticResearchLab/1.0 (https://github.com/arjha10-max/Agentic-Research-Lab)"}
ERRORS = []

# Sources the researcher prefers: market research, consulting, polling and Canadian business/government.
TRUSTED = [
    "statista.com", "mckinsey.com", "nielseniq.com", "nielsen.com", "deloitte.com", "pwc.com",
    "kpmg.com", "bcg.com", "bain.com", "ipsos.com", "kantar.com", "euromonitor.com", "mintel.com",
    "emarketer.com", "thinkwithgoogle.com", "pewresearch.org", "gallup.com", "hbr.org",
    "statcan.gc.ca", "canada.ca", "elections.ca", "leger360.com", "angusreid.org", "abacusdata.ca",
    "mainstreetresearch.ca", "strategyonline.ca", "marketingmag.ca", "canadiangrocer.com",
    "financialpost.com", "theglobeandmail.com", "cbc.ca", "thecanadianpressnews.ca",
    "ey.com", "accenture.com", "gartner.com", "forrester.com", "ibisworld.com",
    "grandviewresearch.com", "mordorintelligence.com",
]
TRUSTED_SHORT = ["statista.com", "mckinsey.com", "nielseniq.com", "deloitte.com", "ipsos.com",
                 "leger360.com", "angusreid.org", "statcan.gc.ca", "marketingmag.ca", "canadiangrocer.com"]

st.set_page_config(page_title="Agentic Research Lab", page_icon=":material/psychology:", layout="wide")
st.markdown("""<style>
.block-container{padding-top:2rem}
.agent{border-radius:14px;padding:14px 18px;margin-bottom:10px;color:#fff;font-weight:600}
.a1{background:linear-gradient(135deg,#6366f1,#8b5cf6)}
.a2{background:linear-gradient(135deg,#0ea5e9,#06b6d4)}
.a3{background:linear-gradient(135deg,#f59e0b,#ef4444)}
</style>""", unsafe_allow_html=True)


# ---------- LLM helper ----------
def llm(client, system, prompt, max_tokens=2000):
    """Chat call with retries. Free reasoning models can spend the whole token budget thinking and
    return an empty reply, so each empty reply retries with a larger budget."""
    last, budget = None, max_tokens
    for attempt in range(3):
        try:
            r = client.chat.completions.create(model=MODEL, max_tokens=budget, messages=[
                {"role": "system", "content": system}, {"role": "user", "content": prompt}])
            choice = r.choices[0]
            text = choice.message.content
            if text and text.strip():
                return text
            last = RuntimeError("The model returned an empty reply (finish reason: " + str(choice.finish_reason) + ")")
            budget = min(budget * 2, 8000)
        except (RateLimitError, APIConnectionError, APITimeoutError, InternalServerError) as e:
            last = e
        time.sleep(3 * (attempt + 1))
    raise last


# ---------- Research tools ----------
# Each tool returns a list of sources: {source, title, text, url}. On any failure it records the
# reason in ERRORS and returns [] so one flaky service never breaks the run.
def allowed(url):
    """True only for the trusted research, consulting, polling and Canadian business/government domains."""
    d = domain_of(url)
    return any(d == t or d.endswith("." + t) for t in TRUSTED)


def domain_of(url):
    return urlparse(url).netloc.replace("www.", "") or "web"


def tavily_call(q, key, domains, recent=True):
    body = {"query": q, "max_results": 4, "search_depth": "basic", "api_key": key}
    if recent:
        body["time_range"] = "year"
    if domains:
        body["include_domains"] = domains
    resp = requests.post("https://api.tavily.com/search", json=body,
                         headers={"Authorization": "Bearer " + key}, timeout=25)
    resp.raise_for_status()
    return [{"source": domain_of(x["url"]), "title": x["title"], "text": x["content"][:1500], "url": x["url"]}
            for x in resp.json().get("results", []) if x.get("content")]


def web_search_tavily(q, key):
    """Trusted marketing and research sites first; open web if those return too little."""
    try:
        rows = tavily_call(q, key, TRUSTED, True)
        if len(rows) < 2:
            rows = tavily_call(q, key, TRUSTED, False)
        return rows
    except Exception as e:
        ERRORS.append(f"Tavily ({type(e).__name__})")
        return []


def web_search_ddg(q):
    """Keyless fallback using DuckDuckGo. Less reliable on shared cloud servers than Tavily."""
    try:
        from ddgs import DDGS
        sites = " OR ".join("site:" + d for d in TRUSTED_SHORT)
        with DDGS() as d:
            rows = list(d.text(q + " (" + sites + ")", region="ca-en", timelimit="y", max_results=8))
            if len(rows) < 2:
                rows = list(d.text(q + " (" + sites + ")", region="ca-en", max_results=8))
        return [{"source": domain_of(r.get("href", "")), "title": r.get("title", ""),
                 "text": (r.get("body") or "")[:1500], "url": r.get("href", "")}
                for r in rows if r.get("href") and r.get("body")]
    except Exception as e:
        ERRORS.append(f"Web search ({type(e).__name__})")
        return []


def norm(s):
    """Lowercase letters and digits only, single spaces. Makes quote matching ignore punctuation."""
    return " ".join("".join(ch.lower() if ch.isalnum() else " " for ch in s).split())


def quote_in_source(quote, text):
    """True if the quote really appears in the source text. '...' gaps are allowed between fragments."""
    nt = norm(text)
    frags = [norm(f) for f in quote.replace(chr(8230), "...").split("...")]
    frags = [f for f in frags if len(f.split()) >= 3]
    return bool(frags) and all(f in nt for f in frags)


def cited_ids(text):
    """Evidence tags like [E3] or [E3, E5] found in text."""
    ids, i = set(), 0
    while True:
        i = text.find("[", i)
        if i < 0:
            break
        j = text.find("]", i)
        if j < 0:
            break
        for part in text[i + 1:j].split(","):
            p = part.strip()
            if p[:1] in ("E", "e") and p[1:].isdigit():
                ids.add(int(p[1:]))
        i = j + 1
    return ids


def numbers_in(text):
    """Numbers in text, ignoring anything inside [brackets]. Commas removed: 1,053 -> 1053."""
    clean, depth = [], 0
    for ch in text:
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth = max(0, depth - 1)
        elif depth == 0:
            clean.append(ch)
    return [n.replace(",", "") for n in re.findall(r"[0-9]+(?:[,.][0-9]+)*", "".join(clean))]


def fact_check(final, ledger):
    """Check every number in the one-pager against the evidence quotes. Returns table rows."""
    by_id = {e["id"]: e for e in ledger}
    rows = []
    for line in final.split(NL):
        nums = numbers_in(line)
        if not nums:
            continue
        tags = cited_ids(line)
        for n in nums:
            in_cited = [i for i in sorted(tags) if i in by_id and n in numbers_in(by_id[i]["quote"])]
            elsewhere = [e["id"] for e in ledger if e["id"] not in tags and n in numbers_in(e["quote"])]
            if in_cited:
                status = "Supported by " + ", ".join("E" + str(i) for i in in_cited)
            elif elsewhere:
                status = "In " + ", ".join("E" + str(i) for i in elsewhere[:3]) + " but not cited on this line"
            else:
                status = "NOT FOUND in evidence"
            rows.append({"Number": n, "Status": status, "Line": line.strip()[:140]})
        for t in sorted(tags):
            if t not in by_id:
                rows.append({"Number": "E" + str(t), "Status": "NOT FOUND in evidence (unknown tag)", "Line": line.strip()[:140]})
    return rows


def candidate_quotes(topic, text):
    """Pick up to 3 data-bearing sentences from a source. Copied by code, so they are verbatim by construction."""
    words = [w for w in norm(topic).split() if len(w) > 3]
    sents = [x.strip() for x in re.split(r"(?<=[.!?]) +", text.replace(NL, " ")) if 40 <= len(x.strip()) <= 400]
    scored = []
    for idx, x in enumerate(sents):
        nx = norm(x)
        score = ((2 if any(ch.isdigit() for ch in x) else 0) + (1 if ("%" in x or "$" in x) else 0)
                 + sum(1 for w in words if w in nx.split()))
        scored.append((score, idx, x))
    top = [t for t in sorted(scored, key=lambda t: (-t[0], t[1])) if t[0] >= 2][:2]
    return [t[2] for t in sorted(top, key=lambda t: t[1])]


def extract_evidence(client, topic, angle, notes):
    """Analysis stage. Code copies the quotes word for word from each source, the model only writes
    the insight for each one (and can mark a quote SKIP when it is not relevant)."""
    items, seen_q = [], set()
    for sid, n in enumerate(notes, 1):
        for q in candidate_quotes(topic, n["text"]):
            if norm(q) in seen_q:
                continue
            seen_q.add(norm(q))
            items.append({"id": len(items) + 1, "source_id": sid, "quote": q, "insight": "",
                          "kind": "data" if any(ch.isdigit() for ch in q) else "context",
                          "verified": quote_in_source(q, n["text"]), "skip": False,
                          "source": n["source"], "title": n["title"], "url": n["url"]})
    items = items[:20]
    chunks = [items[i:i + 10] for i in range(0, len(items), 10)]

    def insight_chunk(chunk):
        lines = NL.join(f"E{e['id']}: {e['quote']} (source: {e['source']})" for e in chunk)
        try:
            return llm(client, "You are a data analyst. Follow the reply format exactly.",
                       f"""Topic: {topic}
Question the POV must answer: {angle}

For each numbered quote below, write one sentence (max 25 words) saying what it means for the question.
If a quote is not relevant to the question, write SKIP instead of an insight.
Reply with one line per quote in the form E3: insight. No other text.

Quotes:
{lines}""", 1200)
        except Exception as e:
            return e

    with ThreadPoolExecutor(max_workers=3) as ex:
        replies = list(ex.map(insight_chunk, chunks))
    for chunk, raw in zip(chunks, replies):
        if isinstance(raw, (AuthenticationError, NotFoundError)):
            raise raw
        if isinstance(raw, Exception):
            ERRORS.append(f"Insights ({type(raw).__name__})")
            continue
        by_id = {e["id"]: e for e in chunk}
        for line in raw.split(NL):
            line = line.strip().lstrip("-*0123456789. ").strip()
            if line[:1] not in ("E", "e") or ":" not in line:
                continue
            head, tail = line.split(":", 1)
            head = head.strip()[1:]
            if head.isdigit() and int(head) in by_id:
                tail = tail.strip().strip("*").strip()
                if tail.upper().startswith("SKIP"):
                    by_id[int(head)]["skip"] = True
                else:
                    by_id[int(head)]["insight"] = tail
    kept = [e for e in items if not e["skip"]] or items
    for i, e in enumerate(kept, 1):
        e["id"] = i
        e["insight"] = e["insight"] or "(no insight returned by the model)"
    return kept


# ---------- Sidebar ----------
with st.sidebar:
    st.title("Settings")
    st.selectbox("Provider", ["OpenRouter (free models)"])
    BASE_URL = "https://openrouter.ai/api/v1"
    MODEL = st.text_input("Model (must end in :free)", "nvidia/nemotron-3-super-120b-a12b:free")
    akey = st.text_input("API key", os.getenv("LLM_API_KEY", ""), type="password").strip()
    tkey = st.text_input("Tavily API key (recommended, free tier)", os.getenv("TAVILY_API_KEY", ""),
                         type="password").strip()
    st.caption("Tavily searches marketing, consulting, polling and Canadian business sources. "
               "Without a key the app falls back to a keyless web search, which is less reliable.")
    st.markdown("---")
    st.markdown(NL.join(["**Pipeline**", "1. Plan (instant)", "2. Research (parallel)", "3. Analysis (parallel)", "4. Writer", "5. Fact-check (instant)"]))

st.title("Agentic Research Lab")
st.caption("Type a topic. Get a one-page point of view built from marketing, consumer and polling sources.")
topic = st.text_input("Topic", "consumer acceptance of zero sugar drinks in Canada")
st.caption("Examples: elections in Canada | how Lunar New Year is celebrated in Canada | EV adoption in Canada")
go = st.button("Build one-pager", type="primary", disabled=not (akey and topic))

if go:
    try:
        ERRORS.clear()
        client = OpenAI(api_key=akey, base_url=BASE_URL)
        t0, S = time.time(), st.session_state
        prog = st.progress(0, "Starting...")
        year = time.localtime().tm_year

        # Plan is built in code, so there is no AI call and no waiting.
        plan = {"angle": f"What do the latest data say about {topic}, and what should decision makers do about it?",
                "queries": [f"{topic} statistics report {year}", f"{topic} consumer survey attitudes",
                            f"{topic} market trends", f"{topic} research insights {year - 1}"]}
        prog.progress(10, "Researching...")

        with st.status("Research Agent: searching trusted sources in parallel...", expanded=False) as s:
            search = (lambda q: web_search_tavily(q, tkey)) if tkey else web_search_ddg
            with ThreadPoolExecutor(max_workers=4) as ex:
                results = [[n for n in r if allowed(n["url"])] for r in ex.map(search, plan["queries"])]
            seen, notes = set(), []
            for i in range(max((len(r) for r in results), default=0)):
                for r in results:
                    if i < len(r) and r[i]["url"] and r[i]["url"] not in seen:
                        seen.add(r[i]["url"])
                        notes.append(r[i])
            notes = notes[:8]
            s.update(label=f"Research Agent: {len(notes)} sources gathered", state="complete")
        if not notes:
            raise RuntimeError("No results came back from the trusted research sources (Statista, McKinsey, Nielsen, Deloitte, Ipsos, Statistics Canada and similar). Add a free Tavily key in the sidebar, or try a more specific topic.")
        prog.progress(40, "Analysing...")

        with st.status("Analysis Agent: pulling quotes and writing insights in parallel...", expanded=False) as s:
            ledger = extract_evidence(client, topic, plan["angle"], notes)
            good = [e for e in ledger if e["verified"]]
            s.update(label=f"Analysis Agent: {len(good)} of {len(ledger)} quotes verified in their sources",
                     state="complete")
        if len(good) < 3:
            raise RuntimeError(f"Only {len(good)} usable quotes were found in {len(notes)} sources. "
                               "Try a more specific topic, add a Tavily key, or run again.")
        prog.progress(65, "Writing...")
        facts = NL.join(f"[E{e['id']}] {Q}{e['quote']}{Q} ({e['source']}) | Insight: {e['insight']}" for e in good)

        with st.status("Writer: drafting the one-pager from the evidence...", expanded=False) as s:
            final = llm(client, "You write one-page point-of-view briefs for marketing and strategy leaders. "
                                "Plain, confident and specific.",
                        f"""Topic: {topic}
Question to answer: {plan['angle']}

Evidence ledger. Every fact you use must come from here and carry its tag, like [E4]:
{facts}

Write a one-page POV of 300 to 400 words in markdown with exactly these parts:
## The POV
2 to 3 sentences that take a clear position, not a summary.
## What the evidence says
4 to 5 bullets. Each bullet holds one specific fact or number and ends with its evidence tag.
## Why it matters
3 bullets on what this means for brands, marketers or decision makers. Tag any fact you rely on.
## Watch-outs
2 bullets on gaps, risks or things to verify.

Rules: use only facts that appear in the ledger. Never invent numbers, dates or quotes.
Copy numbers exactly as written in the quote. If the ledger lacks data on something, say so
under Watch-outs. Do not write a sources list.""", 3000)
            s.update(label="Writer: one-pager complete", state="complete")
        prog.progress(100, "Done")
        S.result = dict(errors=sorted(set(ERRORS)), plan=plan, notes=notes, ledger=ledger,
                        final=final, secs=round(time.time() - t0, 1))
    except AuthenticationError:
        st.error("The API key was rejected (401). Create a new key at openrouter.ai/keys and paste the whole key with no spaces.")
    except NotFoundError:
        st.error("Model not found. Use a model name that ends in :free and is listed on openrouter.ai/models.")
    except RateLimitError:
        st.error("The free model is rate-limited or busy. Wait a minute and click Build one-pager again, or try another :free model.")
    except Exception as e:
        st.error(f"Workflow failed: {type(e).__name__}: {e}")

if "result" in st.session_state:
    R = st.session_state.result
    ledger = R["ledger"]
    by_id = {e["id"]: e for e in ledger}
    checks = fact_check(R["final"], [e for e in ledger if e["verified"]])
    bad = [c for c in checks if c["Status"].startswith("NOT FOUND")]
    used = sorted(i for i in cited_ids(R["final"]) if i in by_id)
    sources_md = NL.join(f"- **E{i}** [{by_id[i]['title']}]({by_id[i]['url']}) ({by_id[i]['source']}): {Q}{by_id[i]['quote']}{Q}"
                         for i in used)
    full = R["final"] + NL * 2 + "### Evidence behind this page" + NL + (sources_md or "No evidence tags found.")

    buf = io.StringIO()
    wr = csv.writer(buf)
    wr.writerow(["id", "verified", "kind", "source", "title", "url", "quote", "insight"])
    for e in ledger:
        wr.writerow([f"E{e['id']}", e["verified"], e["kind"], e["source"], e["title"], e["url"], e["quote"], e["insight"]])

    c = st.columns(4)
    c[0].metric("Sources gathered", len(R["notes"]))
    c[1].metric("Evidence verified", f"{sum(1 for e in ledger if e['verified'])} of {len(ledger)}")
    c[2].metric("Numbers not in evidence", len(bad))
    c[3].metric("Runtime (s)", R["secs"])

    t = st.tabs(["One-pager", "Analysis", "Fact-check", "Research", "Plan"])
    with t[0]:
        st.markdown(full)
        st.download_button("Download one-pager (.md)", full, "one_pager.md")
    with t[1]:
        st.markdown('<div class="agent a1">Analysis Agent: evidence ledger</div>', unsafe_allow_html=True)
        st.caption("Quotes are copied by code, word for word, from the source text and re-checked against it. The AI only writes the insight.")
        for e in ledger:
            flag = "Verified in source" if e["verified"] else "NOT found word for word, excluded"
            st.markdown(f"**E{e['id']}** | {e['kind']} | {flag} | {e['source']}")
            st.markdown("> " + e["quote"])
            st.markdown(f"Insight: {e['insight']}  " + NL + f"[Open source]({e['url']})")
            st.divider()
        st.download_button("Download evidence ledger (.csv)", buf.getvalue(), "evidence_ledger.csv")
    with t[2]:
        st.caption("Every number in the one-pager is looked up in the verified evidence quotes.")
        if checks:
            st.dataframe(checks, hide_index=True)
        else:
            st.write("The one-pager contains no numbers to check.")
    with t[3]:
        st.markdown('<div class="agent a2">Research Agent</div>', unsafe_allow_html=True)
        if R.get("errors"):
            st.caption("Notes from this run: " + ", ".join(R["errors"]))
        for i, n in enumerate(R["notes"], 1):
            with st.expander(f"[{i}] {n['source']} - {n['title']}"):
                st.write(n["text"])
                st.markdown(f"[Open source]({n['url']})")
    with t[4]:
        st.write("**Question the POV answers**")
        st.write(R["plan"]["angle"])
        st.write("**Search queries**")
        for q in R["plan"]["queries"]:
            st.code(q)
