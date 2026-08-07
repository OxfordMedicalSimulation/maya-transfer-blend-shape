# Transfer Blend Shape — testing detached shell support

Branch `fix/detached-shell-support`, pull request #3.

For heads built as a face skin plus separate eyebrow cards and eyelash strips.
Those pieces used to collapse to the origin, or come through over-deformed.
This is what changed, and what to check.

## What actually changed

The tool solves the face skin exactly as it always has. Anything that is a
*separate piece of geometry* — brow cards, lash strips — can't take part in that
solve, so it's now lifted out and moved to match the solved face instead. You
choose, per piece, whether it keeps its shape and simply travels with the face,
or bends along with it.

Nothing else about the workflow moves. You still set source and target the same
way, and you still get one mesh per shape. The new controls sit between
*Colour sets* and the two transfer buttons.

## Working through it

**1. Set source and target as normal.** No change here.

**2. Press *Analyse shells*.**

The list stays empty until you do. You'll get a summary like `3 shells, 2
followed` and one row per piece of geometry.

> Expected: the list clears every time you set the source, target or virtual
> mesh. Your follow modes aren't lost — press *Analyse shells* again and they'll
> be as you left them.

**3. Work out which row is which, using *Select*.**

Rows are numbered by size, not named, so shell 1 could be a brow or a lash.
*Select* puts that piece's vertices in the viewport so you can see it.

Row 0 is the largest piece — normally the face skin. It has no dropdown, only
the word `solved`, because it goes through the ordinary solve. That's correct,
not a missing control.

**4. Set the follow mode on every remaining row.** Three options, which are
three points on one scale — how much the piece resists deforming.

**5. Transfer as normal** — *Transfer selection* or *Transfer from blend shape*,
unchanged.

## The three follow modes

| Mode | Stiffness | What it does | Use for |
|---|---|---|---|
| **Rigid** | 1.0 | Keeps the piece's shape **exactly**. It still moves and rotates to track the face — it just never bends. | Eyebrow cards |
| **Half** | 0.5 | Midway between the two. | When neither end looks right |
| **Smooth** | 0.0 | Lets the piece bend with the skin beneath it, so it stays sitting on a lid that changes shape. | Eyelash strips |

**Rigid does not mean "stays still".** It means "doesn't deform". A Rigid brow
card still follows the face — that's the whole point. This is the label most
people read the wrong way round.

Rigid is the default for every piece, so if you change nothing you get rigid
cards *and* rigid lashes.

## The "Source offset" checkbox

Leave it on — that's the intended setting, and it now shows as ticked when you
open the window.

What it does, if you're curious: with it on, a shape where the artist
deliberately pushed a brow card *further* than the skin underneath keeps that
extra push. With it off, the card only ever does what the skin does.

> Version check: in earlier builds this box displayed unticked while the feature
> was actually on, and took two clicks to turn off. That's fixed on this branch.
> If you see it unticked when the window opens, you're on an older build — tell
> us before testing further.

## What to test

- Set source and target, press *Analyse shells*. Does the shell count match what
  you'd expect for that head — one face, plus your brows and lashes?
- Is row 0 really the face? Confirm with *Select*. If the head is split unusually
  it might not be, and we need to know.
- Leave the brows on **Rigid**. Set the lashes to **Smooth**.
- Transfer one shape that moves the brows, and one blink.

## Known limitations — please don't report these

Two behaviours are understood and deliberately left alone. Fixing either means
reworking maths that currently works, and the judgement was that it isn't worth
the risk for the improvement it would buy.

**Brows come through slightly too far, or not quite far enough.** Which way it
goes depends on the shape. Carrying an authored offset from one head to another
leaves a small error that can land either side of correct — around ten percent
on test geometry. Expect to correct brow shapes by hand.

**Eyelids over-correct on heavy brow shapes.** This is the original solver, not
the new detached-shell handling; the previous version does exactly the same. If
it's bad on a particular shape, try lowering **Iterations** (default 3) — that
controls the smoothing pass driving it, and costs nothing to experiment with.

## What to report

- Anything sitting away from the head — at the origin or otherwise
- Brow cards changing shape: bending, stretching, skewing. They should be
  perfectly rigid.
- Lashes tearing, spiking, or stair-stepping along the lid
- Lashes peeling off a lid that's changing shape
- A shape where the brows move that plainly shouldn't touch them at all — as
  opposed to moving by the wrong amount, which is the known limitation above
- A shell count or ordering that doesn't match the geometry

Worth including: which head, which shape, the shell count and row numbers from
the list, and the follow mode you had on each row.

## Open questions

Two things nobody has been able to check without real geometry, so your answers
here are the point of the exercise.

- **Is the largest shell always the face?** If a head is split differently, row 0
  may not be the skin.
- **How do lashes read on Smooth versus Half?** If Smooth looks wrong, try Half
  and say which you preferred.
