# Intuition in a bottle: hackathon handoff

Nov 24, 2026 · @Krit

We're building a GTM engine for founders breaking into a market: it captures the founder's gut from how they talk, combines it with the whole market's public data, and shows them who needs what they built. The pitch in one line: the whole market's data + one founder's gut = unbeatable GTM.

## The thesis

Founders breaking into a market have a cold-start problem, and their own intuition is the data that's missing.

- **Cold start.** The recommender systems taught in CS224W assume lots of private interaction data. A startup with five customers has almost none.
- **What intuition is.** Herbert Simon called intuition recognition: experience compressed into a fast pattern-matcher. Gary Klein found experts recognize situations rather than compare options.
- **Experts know more than they can say.** Polanyi's point. So we capture intuition through stories about specific customers, not by asking the founder to list rules.
- **Where intuition holds.** Kahneman and Klein (2009): expert intuition is reliable only where the environment is regular and the expert has had feedback. The founder's gut is trustworthy near their experience and untested far from it.
- **Discovery.** The public market graph covers where the gut can't reach. That's where the magic is: use cases and segments the founder never imagined. It's the same reason sponsors pay for hackathons.

**Our definition:** founder intuition is a recognition function the founder learned from experience in their market. It's partly explainable, partly tacit, reliable where they've had real feedback, and unreliable outside it.

**The formula:** the whole market's data + one founder's gut = unbeatable GTM.

## The reading

The founder's first session is a reading, not a dashboard: they talk, and the product shows them who needs what they built.

1. **Enter.** Full screen, no navigation, one button: begin.
2. **Talk.** Voice first. The interviewer asks three or four questions about specific moments, each building on the last answer:
   - "Tell me about the last customer who got really excited. What was happening at their company right before?"
   - "Tell me about a deal you lost that you thought you'd win."
   - "What were people doing before they found you?"

   As they talk, a few of their key phrases quietly settle on screen. No scrolling transcript.
3. **The pause.** "Let me look." The real search runs while company names flicker past like shuffled cards. Under ten seconds.
4. **Understood.** Their own words reflected back, then the hit: a pattern in their named customers they never mentioned. For example, all three hired a head of RevOps within six months of signing.
5. **Surprised.** "The one you didn't see": a segment that describes the founder's pain in different words, with eight or more companies and their own quotes.
6. **Needed.** A stranger's real post describing the founder's problem, dated and linked, then the other companies in that same moment.
7. **The blind spot.** "Here's where I can't see clearly yet."
8. **The next step.** One company, one reason, one email in the founder's voice.

Then the dashboard with the brief's modules opens behind it. When the founder returns, they get a short reading of what changed: who replied, and how buyers describe the pain versus how the founder does.

## How it should feel

A fortune teller with receipts: the founder feels understood and reassured, and every claim survives a click.

**The reading**

- Each card shows the statement on the front and the evidence on the back: dated posts, the lift, the odds it's chance.
- Confidence shows as clarity: strong findings in solid type, weak ones visibly hazier. No percentages on the front.
- It listens mid-reading. "Yes, exactly" strengthens a finding; "no" revises the card in place and says what changed.
- No mystical costume: no crystal balls or purple stars. Calm and editorial.

**Hospitality**

- Dialogue, not monologue (Danny Meyer). The brief's modules are service; the reading is hospitality.
- Read the table: an anxious founder with no customers gets reassurance and one next step; one with traction gets expansion.
- The unreasonable gesture (Will Guidara): a detail mentioned in passing comes back later, usefully.
- Recovery: when corrected, own it visibly. A well-handled mistake builds more trust than none.
- Open kitchen: the Band agent feed and the card backs show the rigor; the founder never has to work in it.
- Peak and end (Kahneman): the peak is the discovery; the end is a real reply arriving.

**Sincerity**

- Never flatter. Being willing to say what they don't want to hear makes everything else believable.
- Five companies, not five hundred. Refusing volume is our costly signal in a spammy category.
- Sending has weight: "You're about to write to the head of data at Acme. Here's why she'd want to hear from you."
- Remember everything; never ask the same thing twice.

**Voice**

- Card copy: second person, present tense, under 20 words, numbers and the founder's own words. No adjectives, no "great insight".
- Emails sound like the founder: their own sent emails as examples, plus an AI-sounding check before the approval screen.
- One person has the final say on taste for every screen.

## How it works

&#91;embedded content: the stack · two inputs, one insight engine, two outputs, one loop\]

Public data is precomputed into the graph and model; the founder's interview fills the bottle live. The insight engine combines both into the reading and the GTM engine, and replies flow back to update the bottle's trust.

## The math that keeps it honest

Every card the founder sees comes from one of these computations, so the magic is checkable.

**Ranking: market plus founder.** Blend the market model with the founder's signal, weighted by how close company c is to the founder's experience set E (their customers, lost deals, named companies):

```latex
s(c) = \hat{y}_{uc} + \lambda(c)\, s_{\text{founder}}(c), \qquad \lambda(c) = \lambda_0\, e^{-d(c,\mathcal{E})/\tau}
```

Near the founder's home turf, their gut dominates. Far from it, the market data takes over.

**Market model training.** Companies that adopted a tool should outscore random companies that didn't (BPR loss):

```latex
\mathcal{L}_{\text{BPR}} = -\sum_{(u,\, i^{+},\, i^{-})} \log \sigma\big(\hat{y}_{u i^{+}} - \hat{y}_{u i^{-}}\big)
```

**Cold start.** A new product starts from its description, then folds in its few known customers with every other embedding frozen:

```latex
\mathbf{e}^{(0)}_{\text{new}} = g_\phi(\text{description}), \qquad \mathbf{e}^{*} = \arg\min_{\mathbf{e}} \sum_{(r,t) \in \mathcal{T}_{\text{new}}} \mathcal{L}(\mathbf{e}, \mathbf{r}, \mathbf{t})
```

**The hit.** Show a pattern among the founder's customers only if it's rare in the market, very unlikely to be chance, and never mentioned in the interview:

```latex
\text{lift}(f) = \frac{P(f \mid \text{customers})}{P(f \mid \text{market})}
```

If 5% of the market has feature f and all 3 customers have it, the odds of chance are 0.05 × 0.05 × 0.05, about 0.0001.

**Discovery.** Score each segment by how well it fits and how far it sits from what the founder already knows:

```latex
\text{discovery}(S) = \text{fit}(S) \cdot d(S, \mathcal{E})
```

**Pain match.** How closely a company's own posts describe the founder's pain, even in different vocabulary:

```latex
\text{pain}(c) = \max_{p \in \text{posts}(c)} \cos(\mathbf{v}_{\text{founder}}, \mathbf{v}_p)
```

**Trust and credits.** Each founder rule k earns trust from outcomes, q\_k \~ Beta(α\_k, β\_k), updated whenever a matching account replies or doesn't. Enrichment spends Flexprice credits in order of p\_i / c\_i (calibrated probability over enrichment cost) until the balance hits zero.

## Proof judges will ask for

Two numbers answer "how do you know it works?", and the second turns the pitch line into a measured result.

1. **Backtest.** Train on data through 2024, predict which companies adopt a given tool in 2025, and compare Recall@50 against an LLM ranking the same candidates from company descriptions. Only test on months after each LLM's training cutoff, or the baseline is remembering, not forecasting.
2. **Ablation.** Market model alone vs founder input alone vs both together, on Recall@50. Simulate the founder with the tool's early customers and a short description of the pain it solves. If the combination clearly beats either part, "market data + founder gut" is a result, not a slogan.

For the demo, have a real tool with lots of history play "your product", so the backtest has enough data. Before judges see the hit or the discovery, try them on two or three real founders: "huh, that's true" means it works; a shrug means fix it first.

## Demo script

Three minutes, built backwards from one peak (the discovery) and one ending (a real reply).

1. **Open, 15s.** "Founder-led sales works because founders have intuition about who'll buy. It breaks the moment you go beyond the companies you know, and every GTM tool needs data a new startup doesn't have."
2. **Talk, 45s.** The founder talks in the reading. Key phrases settle on screen, and the bottle fills in at the side: pain, trigger, buyer, where their knowledge runs deep.
3. **The reading, 45s.** The reflection, then the hit. Flip one card to show the receipts. One planned correction ("no, they're too big") and the card revises in place.
4. **Peak: the discovery, 20s.** "The one you didn't see," with a stranger's own words describing the founder's pain.
5. **The engine, 20s.** Ranked accounts with evidence chips, the Flexprice meter spending credits on the top accounts, and the Band feed showing the Reviewer rejecting a bad triple.
6. **Send, 10s.** Approve one email written in the founder's voice.
7. **End: the reply, 10s.** A teammate replies from their phone, the webhook fires, and the card moves to Replied on its own.
8. **Close, 15s.** The backtest and ablation numbers, then the line, then stop: "The whole market's data + one founder's gut = unbeatable GTM."

Optional: run it on a sponsor's product, but only if its discovery is genuinely non-obvious. Record a backup video of the full run.

## Who builds what

Data and the market model are the critical path: every card in the reading is a query against them.

| Layer | What's in it | Owner | Priority |
| --- | --- | --- | --- |
| Data | HN "Who is hiring" posts, case studies, Wikidata; LLM extraction with citations; entity resolution; Neo4j with the brief's schema; post embeddings | ML teammate | Critical path |
| Market model | Recommender over company–tool adoption (LightGCN or RotatE), description-to-embedding map, fold-in | ML teammate | Critical path |
| The bottle | Voice interview, speech-to-text, pain / trigger / buyer extraction, rule grounding, experience set, trust | Krit | Core |
| Insight engine | Hit finder, discovery, pain matcher, blind spots; each returns a card with receipts | ML teammate computes; Krit writes the card voice and reveal | Core |
| GTM engine | Ranked accounts, Reviewer agent checking citations, enrichment gated by Flexprice, approve and send, reply webhook, Kanban, analytics | Krit | Thin but working |
| Integrations | Snyk in CI, Band agents, live streaming to the UI | Krit | Snyk first |
| Experience | The reading, Accounts, Bottle inspector, Outreach queue, Pipeline | Krit | Where we win the room |

The brief makes two things mandatory: source citations on every prospect, and human approval before any message goes out. Cut: multi-hop query reasoning (Query2Box), R-GCN, LinkedIn sending, multi-channel sequences, and any graph view fancier than one evidence path.

## How it stands out

Every GTM tool reads your data. We read your founder.

- Existing tools run on website traffic, product users or CRM history (Common Room, Unify, Actively, Landbase). A founder breaking into a market has none of that.
- Landbase predicts a company's markets from its URL. Our answer: a website is marketing copy anyone can scrape; the founder's knowledge of who buys and why isn't written down anywhere.
- Actively replicates enterprises' best sellers from their own GTM history. Startups don't have that history, but they do have a founder.
- Their feeds are queues of work. Ours starts with a reading, finds markets the founder hasn't imagined, and sends five emails, not five hundred.

Full comparison: Doc

## Open questions

Settle these before writing code; the first one decides the whole schedule.

- [ ] Do the rules allow building the data and the market model before the event? If not, the first hours are data.
- [ ] Where's the judging rubric (hackathon page or Discord)? It decides how thin the outreach and pipeline modules can be.
- [ ] Which real tool plays "your product" in the backtest?
- [ ] Whose sent emails give the demo founder their voice?
- [ ] Have two or three real founders tried the hit and the discovery?
- [ ] Does the engine find something genuinely non-obvious for Band or Flexprice? If not, drop the sponsor run.
- [ ] Does voice input and output hold up in a noisy room?

After the hackathon: a fine-tuned model of the founder's voice, and research on using a knowledge graph structurally (SALT-KG) with missing edges treated as unlabeled rather than negative.
