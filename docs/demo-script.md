# Demo talk track (3 minutes)

The names and numbers below are what the app shows with seed 7. If you change the seed, run the demo once and update them.

## Before you go on

- [ ] `make sim`, then `make run`. Open http://localhost:8000 in a full-screen browser window.
- [ ] Run the whole flow once to warm up, then reload the page so you start on the Begin screen.
- [ ] A teammate has a phone on the same network, ready to open the reply page. You'll show them the link on screen at step 6, or send it to them then.
- [ ] Turn on the mic only if the room is quiet. Otherwise type, or use "fill demo answer".
- [ ] Have the backup video open in another tab.

## The run

**1. Open (15 s).** *Begin screen. Don't click yet.*

> "Founders sell their first customers on instinct. They know who buys and why, but they can't write it down, and every sales tool needs data a new startup doesn't have. We take the founder's instinct and check it against the public market."

**2. Interview (40 s).** *Click Begin.*

Speak or type the first answer yourself. Use "fill demo answer" for the other three and read them aloud quickly.

> "I'm playing the founder of Ledgerlight. It reconciles billing, CRM and the ledger. My customers are Dapplenest, Pebblepath and Lumen Mutual. They were all reconciling Stripe against Salesforce in spreadsheets."

Point at the right-hand panel as it fills in:

> "It pulls out customers, the problem, who buys and the deal we lost. Each one becomes a rule with a trust score."

**3. The reading (50 s).** *Click Next after the last answer. The search screen runs for about 3 seconds.*

> "Now it scores 900 companies against what I said."

- **From the interview:** "First it shows me what it heard." Click Next.
- **Shared customer trait:** "3 of 3 customers hired RevOps in the 6 months before buying. In similar companies, it's 1%. I never mentioned RevOps."
  Click **Evidence**: "Every claim links to the post it came from, with a date. The q-value is corrected for testing every feature, not just this one." Click Next.
- **New segment:** "31 logistics companies post about the same problem. They call it freight audit. I'd never have searched for that."
  Type **"No, they're too big."** and click **Wrong**: "When I push back, it adds a filter and re-ranks, and it tells me exactly what it changed."
- **Public post:** one line on the Ask HN title. Then click through **Low data**: "It also says where it doesn't have enough data to rank."
- **First account:** "Start with Fablesync, the VP Finance." Click **Accounts**.

**4. The engine (25 s).**

> "Five accounts, not five hundred. Every chip is evidence from a post, and a Reviewer agent checks each one before it shows up."

Click **Enrich top accounts**:

> "Enrichment costs Flexprice credits. It spends them in order of score per credit and stops when the budget runs out."

Optional, 5 s: open the **Band feed** tab. "Every agent step shows up here, including the Reviewer rejecting made-up triples. With a Band key set, the same events go to a Band room." (Only say the Band room part if `BAND_API_KEY` is set and you've seen it work.)

**5. Send (15 s).** *Back on Accounts, click **Draft email** on Fablesync.*

> "It drafts in my voice, from my own sent emails, and checks for phrases that sound generated. Nothing sends until I approve it."

Click **Approve and send**.

**6. The reply (10 s).** *Read out or show the reply link that appears under the button. The teammate opens it on their phone and taps Reply.*

> "Our teammate is playing the VP Finance at Fablesync."

The Pipeline tab opens by itself and the card moves to Replied.

> "The reply came back through the webhook, and it raised the trust score on the rules that picked this account."

**7. Model & proof (20 s).** *Click **Model & proof**.*

Say this line as written:

> "This panel is the roadmap. The training ran on synthetic data, the voice fine-tune is simulated, and there are no measured numbers yet."

Then, pointing at the top table:

> "We planted patterns in the synthetic market, and the pipeline found all four without being told."

Pointing at the backtest table:

> "In the backtest, market plus founder beats either one alone on the point estimate. The confidence intervals still overlap, so we don't claim more than that."

**8. Close (5 s).**

> "The market's data plus one founder's instinct. Thanks."

Stop there.

## If something breaks

| Problem | Do this |
|---|---|
| Mic doesn't pick up speech | Type, or press "fill demo answer". Don't debug on stage. |
| Search screen hangs | Reload, click Begin, fill all four answers. The reading takes about 0.3 s. |
| Phone can't reach the laptop | Open `/reply/<draft id>` in a second browser tab and click Reply there. |
| Server down | Switch to the backup video. |

## Questions judges will likely ask

- **"Is this real data?"** No. It's a seeded synthetic market shaped like Hacker News "Who is hiring" posts. The extraction, the model, the statistics and the backtest are real code running on it. The next step is the real HN corpus.
- **"How do you know the RevOps pattern isn't chance?"** It's compared against companies of the same size and stage over the same months, and corrected for testing every feature. It's also one of the patterns we planted, and the pipeline found it.
- **"Doesn't founder bias hurt?"** Sometimes it does. In the Thompson sampling run, the founder's beliefs used as stated did worse than no beliefs. Rescaled to the market base rate, they did best. The panel shows both.
- **"Why not just ask an LLM?"** The plan is to benchmark against one, scored only on months after its training cutoff. That result doesn't exist yet, so the Measured section is empty.
