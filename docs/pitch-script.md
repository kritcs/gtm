# 3-minute pitch

About 420 words, which is roughly 3 minutes spoken. Screen cues are in brackets.

---

**[Front page]**

Early-stage founders have one real edge when they sell: their gut. They know who buys and why, but they can't prove it, and they can't scale it.

**[The problem]**

The data doesn't help much either. A startup's own data is a handful of customers, far too few to learn from. Public data is huge but noisy. And the usual answer, a knowledge graph, only works once you have a lot of data. So neither source works alone.

Our idea is to use the founder's instinct as the starting point and let the market correct it.

**[How it works → start the interview]**

It starts with a four-question interview. We pull out the founder's beliefs about pain, triggers, buyers and segments. How they say something matters. "Every customer" starts as a strong belief, and "sometimes, I think" starts as a weak one. Technically, each belief is a Beta prior, and its strength comes from the founder's wording.

**[The reading]**

Then we test those beliefs against the market. The market is a heterogeneous graph: companies, tools, hiring signals and outcomes, each a different node type with typed edges between them. A LightGCN model, trained with BPR, learns which companies look like they'll adopt. A new product with only three customers gets placed into that graph by a cold-start fold-in, with no full retrain.

We also look for patterns the founder never mentioned. Here, the three customers all hired RevOps right before buying. We compare that against similar companies, not the whole market, and correct for testing many features at once. It comes out at about 1 in 80,000 odds of being luck.

**[Top accounts]**

The ranking blends both sources. The model's score is always there, and the founder's score is weighted more for companies close to what the founder already knows. Far from that, we trust the data more.

**[Results tab]**

We backtested it: train on data through 2024, then see who adopted afterward. Market plus founder found more future adopters than either one alone, and more than ranking by popularity.

**[Draft pop-up → send]**

Outreach is written in the founder's voice. In production, that's a small model fine-tuned on their own emails. Nothing goes out without their approval.

**[Deal stages → reply lands]**

When someone replies, the beliefs that found that account get credit, the model refits, and the list re-ranks. Every reply makes it a little more yours.

**[AI agents tab]**

Each step is an agent, coordinated through Band. Paid data lookups are metered through Flexprice.

**[Close]**

To be upfront: today this runs on a synthetic market we generated, because we didn't have time to collect real data. The architecture is real, and it builds on graph-learning research from Jure Leskovec's group.

Your gut is your best GTM signal. We make it testable, and we make it learn.
