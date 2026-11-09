# Migrations

## What a migration is

A migration translates a V1 configuration into a V2 one. It is a text
operation on a file, and nothing more.

`pipeline_v2/run.py migrate-v1 --input <v1.yaml> --output <v2.yaml>` reads the
V1 file as **data**, writes one new V2 file, and exits. It does not import V1
code, does not call any V1 lifecycle function, does not touch a device, and
does not start training. The command says so in its own output, because a
migration that quietly began a run would be the worst possible outcome of a
configuration edit.

## What a migration is not

**Migration is not adoption.** A migrated config describes an intent. It carries
no evidence: no accepted predecessor run, no checkpoint, no data manifest, no
evaluator manifest, no gate report. A run launched from a migrated config starts
from nothing and can never claim a predecessor.

V1 remains exactly as it is. No V1 file is created, edited, moved, or deleted by
any V2 command. The migration records the V1 file's SHA-256 before and after and
refuses if they differ, so "we did not touch it" is a checked claim rather than
an intention.

## What is deliberately not migrated

Keys with no safe V2 equivalent are reported in the generated file as comments
and listed in the command output. They are not guessed at. In particular a
`pretrained_checkpoint` or `resume_from` is **never** carried across, because
continuing from a checkpoint requires accepted evidence that a translation
cannot manufacture. Promotions are a separate, deliberate step: see the
`features.promotion` block in the S/P/T templates, which stays `null` until a
predecessor has actually been accepted.

## Promotion, when it is allowed

After stage S is accepted, a user may replace the `promotion: null` in
`pilot.yaml` with a block naming the accepted evidence:

```yaml
features:
  promotion:
    predecessor_stage: S
    predecessor_run_id: <accepted run id>
    checkpoint: <accepted checkpoint>
    data_manifest: <accepted data manifest>
    evaluator_manifest: <accepted evaluator manifest>
    gate_report: <accepted gate report>
```

After stage P is accepted, `target.yaml` takes the analogous block with
`predecessor_stage: P`. Every value must be a repository-relative accepted
evidence path or an accepted identifier. A `null` promotion on stage P or T is
a validation error, because a pilot with no accepted smoke run behind it is not
a pilot.

