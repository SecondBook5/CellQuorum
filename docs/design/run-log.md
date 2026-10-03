# Design note: a log file for every run

Status: proposal, not built. Written so the decision can be made with the facts in front of us.

## The gap

Every run folder has a `logs/` directory, and nothing writes to it. When a run goes wrong, what a
person has is:

- the terminal output, which is gone once the window closes and is silent when `run.verbose` is off;
- `provenance/stage_execution_records.json`, which is complete about *what happened to each stage*
  (status, skip reason, error text, warnings) but says nothing about *what the libraries said*;
- stderr from R and other subprocess backends, which the stage keeps only as a truncated excerpt in
  a skip reason.

The lab member debugging a failed run in a year, with nobody to ask, has the first two and not the
third. The things that usually explain a failure (a scanpy warning that a layer was missing, an R
package's message, a retry) are exactly what is not recorded.

## What already exists to build on

- 14 modules already log through the standard `logging` module (`logging.getLogger(__name__)`), so
  their messages are available to anything that attaches a handler to the `cellquorum` logger.
- The `RunReporter` prints stage start and end, the banner and the summary, with three detail levels
  (`quiet`, `normal`, `verbose`).
- The run folder, and `logs/` inside it, are created before any stage runs.
- Each stage already has a structured record, so stage boundaries are known to the executor.

## Proposal

Write one file, `logs/run.log`, for every run, regardless of console verbosity.

**What goes in it**

1. One line per stage event written by the executor: stage started, finished, skipped (with the
   reason), failed (with the exception type and message), each with a timestamp and elapsed time.
2. Every record from the `cellquorum` logger at INFO and above, with its logger name.
3. Python warnings raised during a stage, tagged with the stage that was running.
4. For R and other subprocess backends, stderr in full, written to `logs/<stage>.<script>.stderr.log`
   and referenced from `run.log`. The excerpt in the skip reason stays as it is; the full text is
   what is added.

**What stays out**

- Progress bars and Rich formatting. The file is plain text.
- Anything that is a result. Results belong in `results/` and the provenance records; the log is for
  diagnosis, and nothing should ever have to be parsed out of it.
- Secrets. The log records the configuration *path*, not its contents, and no environment variables.

**Where it plugs in**

- The executor opens the file handler at the start of a run and closes it at the end, in the same
  place it already creates the `RunReporter`. A handler attached to the `cellquorum` logger covers
  every module that already logs, with no change to them.
- A resumed run appends to the existing `run.log`, with a line marking the resume, rather than
  overwriting the history of the run it continues.
- `provenance/artifact_manifest.csv` lists the log files, so a run report can link to them.

## Decisions to make

| Question | Options | Recommendation |
|---|---|---|
| Log level in the file | Follow `run.log_level`, or always INFO and above | Always INFO and above. The file exists so a quiet run can still be diagnosed; letting it follow the console level defeats that. |
| Format | Plain text, or JSON lines | Plain text, one line per event. A person reads it. A JSON sidecar can be added later if the web app needs one. |
| Subprocess stderr | Capture in memory only, or write whole files | Write whole files. They can be large, which is why they get their own files and not lines in `run.log`. |
| Rotation | None, or size-capped | None. A run is a bounded job and its log belongs to that run. |
| Failure behaviour | A logging failure aborts the run, or is swallowed | Swallowed with one warning on the console. Failing to write a diagnostic must never cost someone a result. |

## How it would be tested

- A run of the demo project produces `logs/run.log` containing a start and an end line for every stage
  that ran.
- A stage that raises produces a log entry with the exception type and message, and the run's own
  failure behaviour is unchanged.
- A quiet run (`run.verbose: false`) still writes the file.
- Resuming a run appends and does not truncate.
- An unwritable `logs/` directory yields a warning and a completed run.

## Cost

Small: one handler, one executor hook, one change to how subprocess output is retained. The risk is
in the retention of subprocess stderr, because those files can be large; the size is the thing to
measure on the atlas before turning it on by default.
