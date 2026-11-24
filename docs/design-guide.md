# Design guide: keep it minimal, keep it ours

Read this before changing anything in `apps/web/`. Krit reviews every screen against these rules. Most feedback so far has been one of three things: too much text, text that sounds machine-written, and visuals that don't explain anything.

## 1. Words

**Say less.** A screen gets a title, at most one sentence under it, and the data. If a label repeats on every item, delete it and mark only the exception. For example, rule cards say nothing for the founder's own words and "from market data" only on the one rule the data found.

**Write sentences, not fragment lists.** Never write "Signal, packet, draft" or "Fast · cheap · simple". Write "We check it against what already happened and against companies today." Avoid these:
- Comma or dot-separated strings of noun phrases used as a sentence.
- Em dashes used for drama, and "not X but Y" contrasts.
- Words like *seamless, leverage, unlock, powerful, delve, game-changer*.
- Rhetorical one-line closers.

**Active voice, second person, present tense.** "Every reply moves it up or down", not "Beliefs are updated upon replies".

**Plain English first, then the technical term.** Write "1 in 80,000 odds of luck", not "q = 1.25e-5". Technical terms are welcome where they name the real mechanism (Beta prior, LightGCN, LoRA, Benjamini–Hochberg), but the sentence around them must still read plainly.

**Sentence case everywhere.** Capitalize labels, buttons, legends, tab names and the first word of rule text. Use proper names for industries and stages (SaaS billing, E-commerce, HR tech, Series A); `apps/web/text.js` has `cap`, `prettyInd` and `prettyStage`. In the middle of a sentence, common nouns stay lowercase ("31 logistics companies").

**Don't label the data as synthetic in the UI.** The data is a seeded synthetic market, and Krit says so out loud in the demo. Don't add disclaimers, "Synthetic / Simulated / Measured" badges, or "None yet" placeholders to any screen.

## 2. Type

| Role | Font | Use |
|---|---|---|
| Display and headings | Instrument Serif (`--serif`), weight 400, italic for emphasis words | Page titles, card fronts, company names, big numbers |
| Everything else | Inter (`--sans`) | UI text, labels, buttons, captions |
| Numbers only | Fragment Mono (`--mono`) | Beta(14, 6), weights like +2.82, credit counters. Never labels or sentences. |

All fonts are OFL and live in `apps/web/vendor/`. Don't add Google Fonts links, because the demo must run offline. Headings get `text-wrap: balance`.

## 3. Color

The app background is near-black (`--background: #000`), with cards at `--card: #0e0e0f` and 1px borders at `rgba(255,255,255,.09)`. The front-page flow (front page, The problem, How it works) sits on the indigo-to-cyan field defined on `body.hero-on`, with glass cards over it.

Color always means something. Data colors are the categorical slots, in this fixed order:

| Token | Hex | Means |
|---|---|---|
| `--c1` | `#3987e5` | Your customers, your product, highlighted results |
| `--c2` | `#d95926` | New segment and "different words" matches |
| `--c3` | `#199e70` | Good outcomes: replies, retained |
| — | `#eda100` | A trait or signal being tested |
| — | `#22c55e` / `#ef4444` | Pass / fail in explainers |
| — | `#9085e9` | The model and the product (violet) |

Never introduce a new hue for decoration. If a chart needs more than three series, cut series instead.

## 4. Components

- **Buttons:** pills only (`border-radius: 999px`). Primary is white-filled with black text; the rest are outlined. Prefer an icon plus one word; use icon-only for repeated actions (next, flip, close), with an `aria-label`.
- **Icons:** Lucide, vendored as single SVGs in `apps/web/vendor/icons/`. To add one, run `curl -sfL -o apps/web/vendor/icons/NAME.svg https://unpkg.com/lucide-static@0.460.0/icons/NAME.svg`. Use them through `<i class="i" style="--i:url(/static/vendor/icons/NAME.svg)">`.
- **Cards:** 14–22px radius, 1px border, no drop-shadow on dark surfaces. Use glass cards (`backdrop-filter: blur(10px)`) only on the indigo field.
- **Tables:** avoid them. Every table so far became a visual: gap charts, bars with 95% whiskers, KPI tiles. Use a table only for small lists the viewer must read row by row.
- **Charts:** one y-axis, at most three series, direct labels plus a legend, hover tooltips, and a dashed baseline where one exists (random, popularity). Show uncertainty with whiskers, not ± text.

## 5. Motion

Motion must explain something: a reply moving a card, a belief curve shifting, a scan finding companies. No ambient wiggle for its own sake beyond the slow 3D drift.

- Use easing `cubic-bezier(.2,.8,.2,1)`, with durations of 300–900ms. Stagger lists by 60–110ms.
- Use the helpers in `app.js` (`anim`, `stagger`, `reveal`, `sleep`) and in `explainers.js` (`pop`, `fill`, `fade`).
- **SVG gotcha:** never animate `transform` on an element that has a `transform="translate(...)"` attribute, because it moves the element to the origin. Animate the individual `scale` property instead, as `pop()` does. Use `fill: "backwards"`, not `"both"`, or later opacity changes won't apply.
- Respect `prefers-reduced-motion`. Every helper checks `RM` and runs instantly.

## 6. 3D and backgrounds

- `hero3d.js` draws the front-flow backgrounds in canvas 2D, and each scene carries a meaning. The **bottleneck** (front page) shows market signals funneling through one glowing neck, the founder's instinct, then fanning out to customers. The **fan** (Problem) shows rays from one founder to companies across the market. **`dome3d.js`** draws the arc dome on How it works with three.js: every streak starts at one point (the founder's intuition) and lands on one of the 900 companies. It turns on through `body.dome-on`. Switch scenes with `HeroBG.scene(name)`; they crossfade. The scenes render only while `body.hero-on` is set.
- These scenes take inspiration from Stripe's line art, but they must stay our own: our indigo palette, our meanings, our motion. Never trace or re-create another site's graphic.
- `map3d.js` draws the market point cloud behind the interview and reading. Each card lights up its evidence through `MarketMap.show(card.map)`.
- Both use the vendored three.js r170. Don't load anything from a CDN at runtime.

## 7. Before you commit

1. Bump the `?v=` number on every changed asset in `index.html`, or browsers keep serving the old file.
2. Check the screen at about 800px and at 1440px wide (the browser pane's resize tool works), then reset the viewport.
3. Read the console for errors.
4. Run `uv run pytest -q` and `uv run python scripts/demo_check.py`.
5. Re-read every string you added against section 1. If a line could be cut without losing meaning, cut it.

## 8. Where things live

| File | What |
|---|---|
| `apps/web/index.html` | Stages: enter → problem → intro (How it works) → talk → pause → reading → dash |
| `apps/web/app.js` | Flow, dashboard tabs, charts, motion helpers |
| `apps/web/explainers.js` | How-it-works pop-up scenes (color-block animations) |
| `apps/web/system-view.js` | Architecture tab (Peem's loop, graph, learning) |
| `apps/web/pipeline-graph.js` | Deal stages force graph |
| `apps/web/agents-view.js` | AI agents tab |
| `apps/web/text.js` | Capitalization and plain-English odds helpers |
| `apps/web/styles.css` | Tokens at the top, then one section per screen |

**The problem page (`#problem` in `index.html`)** makes one argument, and each point appears once:
1. A startup's own data is too thin (cold start), and public data is noisy.
2. Traditional knowledge graphs need volume, so neither source works on its own.
3. Our approach: the founder is the prior (Beta priors, and the graph corrects them where the data is strong). The graph is heterogeneous (typed nodes and edges, so a reply credits its signal). It re-fits as feedback arrives (Bayesian updates and fold-in).

`problem-diagram.js` plays those three points in sync with the list. Don't add "instinct" or "intuition" anywhere else on the page.

The copy on the problem page is a placeholder. Replace it with real numbers only once they're sourced. Until then, keep every sentence free of made-up statistics.
