# Demo talk track (3 minutes)

The names and numbers below are what the app shows with seed 7. If you change the seed, run the demo once and update them.

Buttons are icons now. This script names each one by its icon, and every icon shows its name on hover.

| Icon | Does |
|---|---|
| → (dark) | Next |
| wand | fill the demo founder's answer |
| mic | speak |
| eye | flip the card to its evidence |
| ✓ / ✕ | right / wrong (type the correction in the box first) |
| ← (top left of the dashboard) | back to the reading |

## Before you go on

- [ ] `make sim`, then `make run`. Open http://localhost:8000 in a browser window at least 1280 px wide. The reading puts the card on the right and the map on the left, and narrower windows stack them.
- [ ] Run the whole flow once to warm up, then reload the page so you start on the "Who buys next." screen.
- [ ] A teammate has a phone on the same network, ready to open the reply page. You'll show them the link on screen at step 6, or send it to them then.
- [ ] Turn on the mic only if the room is quiet. Otherwise type, or use the wand.
- [ ] Have the backup video open in another tab.

## The run

**1. Open (15 s).** *Front page: "Intuition in a bottle. Your gut feeling is the ultimate GTM." Don't click yet.*

> "Founders win their first customers on instinct. They know who buys and why, but they can't write it down, and every sales tool needs data a new startup doesn't have. That bright point is a founder's intuition. Each streak is a company it could reach."

**2. Interview (35 s).** *Click **Begin**. The problem page appears. Give it one line ("Startups have too little data, and the market's data is noise. So the founder becomes the prior, and the graph learns only the correction."), then click **See how it works**. On "We turn your instinct into your next customer", click **We test your instinct against the market**. Narrate over the animation, then close it.*

> "We score your instinct in both directions. Backward, we replay the past and throw out anything that could be luck. Forward, we scan the market for companies that show the same signals, even when they use different words."

*Click **Start the interview**.* Speak or type the first answer yourself. Use the wand for the other three and read them aloud quickly. Press → after each one.

> "I'm playing the founder of Ledgerlight. My customers are Dapplenest, Pebblepath and Lumen Mutual. They were all reconciling Stripe against Salesforce in spreadsheets."

As each customer's dot lights up blue:

> "As I name customers, it finds them in the market."

**3. The reading (45 s).** *After the last answer, the count runs up while a wave passes through the dots, for about 3 seconds.*

> "Now it scores every company against what I said."

Press → to move between cards. Each card lights up its evidence on the map.

- **01 · heard.** "First it shows me what it heard."
- **02 · pattern.** The numbers count up. "3 of 3 customers hired RevOps in the 6 months before buying. In similar companies, it's 1%. I never mentioned RevOps."
  Press the eye: "Every claim links to the post it came from, with a date. The q-value is corrected for testing every feature, not just this one." Press the eye again to flip back, then →.
- **03 · new segment.** Orange dots light up far from the blue ones, and an arc draws out to them. "31 logistics companies post about the same problem. They call it freight audit. They're over here, nowhere near my customers. I'd never have searched for that."
  Type **"No, they're too big."** and press ✕. The removed companies strike through, then the card rewrites itself. "When I push back, it adds a filter and re-ranks, and it tells me exactly what it changed."
- **04 · public post.** One line on the Ask HN title.
- **05 · low data.** "It also says where it doesn't have enough data to rank."
- **06 · start here.** A ring appears around one dot. "Start with Fablesync, the VP Finance." Press →. The dashboard opens on the Overview: the whole product on one page. Click **Top accounts**.

**4. The engine (20 s).**

> "Five accounts, not five hundred. Every chip is evidence from a post, and a Reviewer agent checks each one before it shows up."

Press **Enrich**. The meter drains one account at a time.

> "Enrichment costs Flexprice credits. It spends them in order of score per credit and stops when the budget runs out."

Optional, 5 s: open the **Agents** tab. "Every agent step shows up here, including the Reviewer rejecting made-up triples." Only add "With a Band key set, the same events go to a Band room" if `BAND_API_KEY` is set and you've seen it work.

**5. Send (15 s).** *Back on **Accounts**, press **Draft** on Fablesync. The email types itself out, then a line sweeps over it.*

> "It drafts in my voice, from my own sent emails. That sweep is a check for phrases that sound generated. Nothing sends until I approve it."

Press **Approve & send**.

**6. The reply (10 s).** *The reply link appears under the button, and then the Pipeline opens by itself. Read the link out or show it. The teammate opens it on their phone and taps Reply.*

> "Our teammate is playing the VP Finance at Fablesync."

The Fablesync card slides into **replied**.

> "The reply came back through the webhook, and it raised the trust score on the rules that picked this account."

**7. The architecture (20 s).** *Open the **System** tab and press **Run the loop**.*

> "This is the architecture, built on graph-learning research. Eight steps: the forward pass finds and contacts the right accounts."

When the green token takes the shortcut from Learn back to Score:

> "Everyone builds the forward pass. We build the backward pass: outcomes rewrite who we target and what we say."

Optional, only if there's time: switch to **learning** and press **+ churned (0)** once.

> "A founder's belief is a prior, and every outcome moves it."

The dots on each step are honest about status: green is running in this demo, amber is partly running, and grey is roadmap.

**8. Proof (15 s).** *Open the **Proof** tab.*

Say this line as written:

> "Everything here is trained on synthetic data. The voice model is a mock-up, and nothing is measured yet."

Then, pointing at the top table:

> "We planted patterns in the synthetic market, and the pipeline found all four without being told."

Pointing at the backtest table:

> "In the backtest, market plus founder beats either one alone on the point estimate. The confidence intervals still overlap, so we don't claim more than that."

**9. Close (5 s).**

> "The market's data plus one founder's instinct. Thanks."

Stop there.

## If something breaks

| Problem | Do this |
|---|---|
| Mic doesn't pick up speech | Type, or press the wand. Don't debug on stage. |
| The dots don't appear (WebGL off) | Keep going. Every card and button still works without the map. |
| Search screen hangs | Reload, press Begin, fill all four answers. The reading takes about 0.3 s. |
| Phone can't reach the laptop | Open `/reply/<draft id>` in a second browser tab and click Reply there. |
| Server down | Switch to the backup video. |

## Questions judges will likely ask

- **"Is this real data?"** No. It's a seeded synthetic market shaped like Hacker News "Who is hiring" posts. The extraction, the model, the statistics and the backtest are real code running on it. The next step is the real HN corpus.
- **"Which parts of the architecture are built?"** The System tab marks each step. Intake, approval and the reply loop run in the demo. The market graph, scoring, enrichment, signal packets, drafting and learning run partly. Person, Competitor, Stripe outcomes and the GNN residual are roadmap.
- **"What is the map?"** Each dot is a company, placed by the model's own embeddings, so companies that use similar tools and describe similar problems sit close together. That's why the logistics segment appears far from the founder's customers.
- **"How do you know the RevOps pattern isn't chance?"** It's compared against companies of the same size and stage over the same months, and corrected for testing every feature. It's also one of the patterns we planted, and the pipeline found it.
- **"Doesn't founder bias hurt?"** Sometimes it does. In the Thompson sampling run, the founder's beliefs used as stated did worse than no beliefs. Rescaled to the market base rate, they did best. The panel shows both.
- **"Why not just ask an LLM?"** The plan is to benchmark against one, scored only on months after its training cutoff. That result doesn't exist yet, so the Measured section is empty.
