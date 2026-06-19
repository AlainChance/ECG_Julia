#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ecg_config_gradio.py — a Gradio interface to load, edit and save an
``ecg_config.json`` file for ECG_Julia.

An ``ecg_config.json`` has up to four top-level sections:

    {
        "setup":     { "copy": [...], "clean": [...] },
        "ECG_Param": { "verbose": 1, ...module-level settings... },
        "param0":    { ...fields of the Init_param instance... },
        "actions":   [ { "Type": "BASIS_ENL", ... }, ... ]   # optional command script
    }

Because the keys present in ``ECG_Param`` and ``param0`` differ from one model
to another, this editor does NOT hard-code a fixed form: it edits those two
blocks as editable key/value tables (arbitrary keys preserved, rows added or
removed). ``setup.copy`` / ``setup.clean`` are edited one entry per line; a copy
entry of the dict form ``{"from": a, "to": b}`` is shown / typed as ``a -> b``.

``verbose`` (an ``ECG_Param`` setting consumed by ECG_Julia) is surfaced as a
dedicated numeric field rather than a table row.

The optional ``actions`` array is edited in its own tab as a table with one row
per action and one column per field of the ECG_Init ``action`` struct (``Type``
is the action NAME, e.g. ``BASIS_ENL``; ``seed`` is an integer or ``null``).
When present in the saved config, ECG_Init uses it as the command script instead
of the legacy ``inout.txt`` action list. If the table is empty, no ``actions``
key is written (so ``inout.txt`` keeps driving the run).

Consistency with the current ECG_Julia: the neural-surrogate feature was removed,
so on load this tool silently drops the obsolete keys
(``do_surrogate``, ``maxiters``, ``num_new_samples``, ``n_iters``, ``n_echos``,
``delta_target``, ``build``) and the deleted ``ECG_Surrogates.jl`` entry from
``setup.copy`` (reporting what it dropped), so saved files match the new code.

Run:  python ecg_config_gradio.py
(then open the printed local URL in a browser)

Released under the MIT License. Copyright (c) 2026 Alain Chancé.
Inspired by SQD_Alain_Gradio.py (github.com/AlainChance/SQD_Alain).
"""

import os
import re
import glob
import json

try:
    import gradio as gr
except ImportError as exc:  # pragma: no cover
    raise SystemExit("Gradio is required. Install it with:  pip install gradio") from exc

SETUP_ARROW = " -> "          # separator displaying a {"from","to"} copy entry
DEFAULT_VERBOSE = 1

# Keys/files removed from ECG_Julia together with the neural-surrogate feature.
DEPRECATED_PARAM_KEYS = {"do_surrogate", "maxiters", "num_new_samples",
                         "n_iters", "n_echos", "delta_target", "build"}
DEPRECATED_COPY = {"ECG_Surrogates.jl"}

# The ECG_Init `action` struct, as edited in the "actions" table. "Type" is the
# action name (matching ECG_Init.action_type); the rest mirror the struct fields.
ACTION_FIELDS = ["Type", "solver_type", "nfa", "nfo", "ntrials", "MaxEnergyEval",
                 "Kstart", "Kstop", "Kstep", "seed", "nlp0", "coeff_nlp"]
ACTION_DEFAULTS = {"Type": "", "solver_type": "G", "nfa": 0, "nfo": 0, "ntrials": 0,
                   "MaxEnergyEval": 0, "Kstart": 0, "Kstop": 0, "Kstep": 1,
                   "seed": None, "nlp0": False, "coeff_nlp": 0.0}
ACTION_TYPES = ["BASIS_ENL", "OPT_CYCLE", "FULL_OPT1", "CHECK", "BASIS_REPL",
                "BASIS_ENL_F90", "OPT_CYCLE_F90"]


# --------------------------------------------------------------------------- #
# Value (de)serialization
# --------------------------------------------------------------------------- #
def value_to_str(v):
    if isinstance(v, bool):
        return "true" if v else "false"
    if v is None:
        return "null"
    return str(v)


def str_to_value(s):
    """Table string -> typed JSON value. Keeps "T"/"F"/"G"/strings as strings."""
    if isinstance(s, (int, float, bool)) or s is None:
        return s
    t = str(s).strip()
    if t == "":
        return ""
    low = t.lower()
    if low == "true":
        return True
    if low == "false":
        return False
    if low in ("null", "none"):
        return None
    if re.fullmatch(r"[+-]?\d+", t):
        return int(t)
    try:
        return float(t)
    except ValueError:
        return t


def _vint(v, default=DEFAULT_VERBOSE):
    try:
        return int(v)
    except (TypeError, ValueError):
        return default


# --------------------------------------------------------------------------- #
# setup.copy / setup.clean helpers
# --------------------------------------------------------------------------- #
def copy_entry_to_line(entry):
    if isinstance(entry, dict):
        return f"{entry.get('from', '')}{SETUP_ARROW}{entry.get('to', '')}"
    return str(entry)


def line_to_copy_entry(line):
    line = line.strip()
    if not line:
        return None
    if SETUP_ARROW in line:
        a, b = line.split(SETUP_ARROW, 1)
        return {"from": a.strip(), "to": b.strip()}
    return line


def list_to_lines(items):
    return "\n".join(copy_entry_to_line(e) for e in (items or []))


def _is_deprecated_copy(entry):
    name = entry if isinstance(entry, str) else (entry.get("from", "") if isinstance(entry, dict) else "")
    return os.path.basename(name) in DEPRECATED_COPY


# --------------------------------------------------------------------------- #
# Table helpers
# --------------------------------------------------------------------------- #
def _rows(df):
    """Key/value rows of a 2-column table -> list of (key, value)."""
    if df is None:
        return []
    if hasattr(df, "values"):
        df = df.values.tolist()
    out = []
    for row in df:
        if not row:
            continue
        key = "" if len(row) < 1 or row[0] is None else str(row[0]).strip()
        val = "" if len(row) < 2 or row[1] is None else row[1]
        if key == "":
            continue
        out.append((key, val))
    return out


def dict_to_table(d):
    return [[k, value_to_str(v)] for k, v in (d or {}).items()]


def _table_rows(df):
    """Full rows of an N-column table -> list of lists."""
    if df is None:
        return []
    if hasattr(df, "values"):
        df = df.values.tolist()
    return [list(r) for r in df if r is not None]


# --------------------------------------------------------------------------- #
# actions table <-> list of action dicts
# --------------------------------------------------------------------------- #
def actions_to_table(actions):
    rows = []
    for a in (actions or []):
        a = a or {}
        rows.append([value_to_str(a.get(f, ACTION_DEFAULTS[f])) for f in ACTION_FIELDS])
    return rows


def table_to_actions(df):
    out = []
    for row in _table_rows(df):
        cells = list(row) + [""] * (len(ACTION_FIELDS) - len(row))
        if str(cells[0]).strip() == "":          # blank Type -> skip the row
            continue
        action = {}
        for i, f in enumerate(ACTION_FIELDS):
            action[f] = str(cells[i]).strip() if f == "Type" else str_to_value(cells[i])
        out.append(action)
    return out


# --------------------------------------------------------------------------- #
# Assemble / parse a full config
# --------------------------------------------------------------------------- #
def assemble_config(copy_text, clean_text, verbose, ecg_rows, p0_rows, actions_rows):
    copy_list = [e for e in (line_to_copy_entry(l) for l in copy_text.splitlines())
                 if e is not None and not _is_deprecated_copy(e)]
    clean_list = [l.strip() for l in clean_text.splitlines() if l.strip()]

    ecg = {k: str_to_value(v) for k, v in _rows(ecg_rows)
           if k != "verbose" and k not in DEPRECATED_PARAM_KEYS}
    ecg = {"verbose": _vint(verbose), **ecg}            # verbose first, from the widget

    p0 = {k: str_to_value(v) for k, v in _rows(p0_rows)
          if k not in DEPRECATED_PARAM_KEYS}

    cfg = {"setup": {"copy": copy_list, "clean": clean_list},
           "ECG_Param": ecg, "param0": p0}

    actions = table_to_actions(actions_rows)
    if actions:                                         # omit when empty -> legacy inout.txt
        cfg["actions"] = actions
    return cfg


def _load_path(path):
    cfg = json.load(open(path, encoding="utf-8"))
    notes = []

    setup = cfg.get("setup", {}) or {}
    copy_in = setup.get("copy", []) or []
    copy_kept = [e for e in copy_in if not _is_deprecated_copy(e)]
    if len(copy_kept) != len(copy_in):
        notes.append("dropped ECG_Surrogates.jl from setup.copy")
    clean_list = setup.get("clean", []) or []

    ep = dict(cfg.get("ECG_Param", {}) or {})
    verbose = _vint(ep.pop("verbose", DEFAULT_VERBOSE))
    dep_ep = [k for k in list(ep) if k in DEPRECATED_PARAM_KEYS]
    for k in dep_ep:
        ep.pop(k)

    p0 = dict(cfg.get("param0", {}) or {})
    dep_p0 = [k for k in list(p0) if k in DEPRECATED_PARAM_KEYS]
    for k in dep_p0:
        p0.pop(k)

    if dep_ep or dep_p0:
        notes.append("dropped deprecated keys: " + ", ".join(dep_ep + dep_p0))

    actions_in = cfg.get("actions", []) or []
    if actions_in:
        notes.append(f"loaded {len(actions_in)} action(s)")

    clean_cfg = {"setup": {"copy": copy_kept, "clean": clean_list},
                 "ECG_Param": {"verbose": verbose, **ep}, "param0": p0}
    if actions_in:
        clean_cfg["actions"] = actions_in
    preview = json.dumps(clean_cfg, indent=4, ensure_ascii=False)
    return (list_to_lines(copy_kept), "\n".join(str(x) for x in clean_list),
            verbose, dict_to_table(ep), dict_to_table(p0),
            actions_to_table(actions_in), preview, notes)


# --------------------------------------------------------------------------- #
# Filesystem helpers
# --------------------------------------------------------------------------- #
def list_json_files(directory):
    d = directory or "."
    try:
        return sorted(os.path.basename(p) for p in glob.glob(os.path.join(d, "*.json")))
    except OSError:
        return []


# --------------------------------------------------------------------------- #
# Gradio callbacks
# --------------------------------------------------------------------------- #
def refresh_files(directory):
    return gr.update(choices=list_json_files(directory))


def _load_common(path):
    """Returns the 9-tuple matching load_outputs, or raises."""
    copy_text, clean_text, verbose, ecg, p0, actions_tbl, preview, notes = _load_path(path)
    msg = f"Loaded {path}"
    if notes:
        msg += "  |  " + "; ".join(notes)
    return (copy_text, clean_text, verbose, ecg, p0, actions_tbl,
            os.path.basename(path), msg, preview)


def load_selected(directory, filename):
    if not filename:
        return (gr.update(),) * 7 + ("Select a file first.", gr.update())
    path = os.path.join(directory or ".", filename)
    try:
        return _load_common(path)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        return (gr.update(),) * 7 + (f"Error loading {path}: {exc}", gr.update())


def load_uploaded(fileobj):
    if not fileobj:
        return (gr.update(),) * 7 + ("No file uploaded.", gr.update())
    path = fileobj if isinstance(fileobj, str) else getattr(fileobj, "name", None)
    try:
        return _load_common(path)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        return (gr.update(),) * 7 + (f"Error loading {path}: {exc}", gr.update())


def preview_config(copy_text, clean_text, verbose, ecg_rows, p0_rows, actions_rows):
    try:
        cfg = assemble_config(copy_text, clean_text, verbose, ecg_rows, p0_rows, actions_rows)
    except Exception as exc:                       # noqa: BLE001
        return "", f"Error assembling config: {exc}"
    return json.dumps(cfg, indent=4, ensure_ascii=False), "Assembled OK."


def save_config(directory, filename, copy_text, clean_text, verbose, ecg_rows, p0_rows, actions_rows):
    name = (filename or "").strip()
    if not name:
        return "Please provide a filename.", gr.update()
    if not name.lower().endswith(".json"):
        name += ".json"
    d = directory or "."
    try:
        os.makedirs(d, exist_ok=True)
        cfg = assemble_config(copy_text, clean_text, verbose, ecg_rows, p0_rows, actions_rows)
        path = os.path.join(d, name)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(cfg, fh, indent=4, ensure_ascii=False)
            fh.write("\n")
    except Exception as exc:                       # noqa: BLE001
        return f"Error saving: {exc}", gr.update()
    return f"Saved {path}", gr.update(choices=list_json_files(d), value=name)


# --------------------------------------------------------------------------- #
# UI
# --------------------------------------------------------------------------- #
def build_ui(default_dir="."):
    with gr.Blocks(title="ECG_Julia — ecg_config.json editor") as demo:
        gr.Markdown(
            "# ECG_Julia — `ecg_config.json` editor\n"
            "Load an existing configuration, edit the sections "
            "(`setup`, `ECG_Param`, `param0`, `actions`), preview the assembled JSON, and save.\n\n"
            "*Tips:* in **setup.copy**, write one entry per line; a `{\"from\": a, "
            "\"to\": b}` entry is written as `a -> b`. On load, obsolete neural-surrogate "
            "keys and the deleted `ECG_Surrogates.jl` copy entry are dropped automatically "
            "(see the status line). The **actions** tab is optional — leave it empty to keep "
            "driving the run from `inout.txt`."
        )

        with gr.Row():
            dir_tb = gr.Textbox(value=default_dir, label="Directory", scale=3)
            file_dd = gr.Dropdown(choices=list_json_files(default_dir),
                                  label="Config file (*.json)", scale=4)
            refresh_btn = gr.Button("↻ Refresh", scale=1)
            load_btn = gr.Button("Load", variant="primary", scale=1)
        with gr.Row():
            upload = gr.File(label="…or upload a JSON file", file_types=[".json"],
                             type="filepath")

        with gr.Tab("setup"):
            copy_tb = gr.Textbox(label="copy  (one path per line; `from -> to` for a "
                                       "{from,to} entry)", lines=10)
            clean_tb = gr.Textbox(label="clean  (one filename per line)", lines=5)

        with gr.Tab("ECG_Param"):
            verbose_num = gr.Number(value=DEFAULT_VERBOSE, precision=0,
                                    label="verbose  (ECG_Param verbosity level, e.g. 0 / 1 / 2)")
            ecg_df = gr.Dataframe(headers=["key", "value"], datatype=["str", "str"],
                                  col_count=(2, "fixed"), row_count=(1, "dynamic"),
                                  label="ECG_Param — other module-level settings "
                                        "(verbose is the field above)")

        with gr.Tab("param0"):
            p0_df = gr.Dataframe(headers=["key", "value"], datatype=["str", "str"],
                                 col_count=(2, "fixed"), row_count=(1, "dynamic"),
                                 label="param0 (Init_param fields)")

        with gr.Tab("actions"):
            gr.Markdown(
                "Optional command script — one **row per action**. `Type` is the action "
                "name (`" + "`, `".join(ACTION_TYPES) + "`); `seed` is an integer or `null`; "
                "`nlp0` is `true`/`false`. When non-empty, this replaces the `inout.txt` "
                "action list (the system + basis still come from `inout.txt`). Add or remove "
                "rows with the table controls; leave empty to use `inout.txt`."
            )
            actions_df = gr.Dataframe(
                headers=ACTION_FIELDS,
                datatype=["str"] * len(ACTION_FIELDS),
                col_count=(len(ACTION_FIELDS), "fixed"),
                row_count=(1, "dynamic"),
                label="actions (Type, solver_type, nfa, nfo, ntrials, MaxEnergyEval, "
                      "Kstart, Kstop, Kstep, seed, nlp0, coeff_nlp)")

        with gr.Row():
            preview_btn = gr.Button("Preview / validate JSON")
            save_name = gr.Textbox(value="ecg_config.json", label="Save as", scale=2)
            save_btn = gr.Button("Save", variant="primary", scale=1)

        status = gr.Textbox(label="Status", interactive=False)
        preview = gr.Code(label="Assembled JSON (preview)", language="json")

        # Wiring
        load_outputs = [copy_tb, clean_tb, verbose_num, ecg_df, p0_df, actions_df,
                        save_name, status, preview]
        edit_inputs = [copy_tb, clean_tb, verbose_num, ecg_df, p0_df, actions_df]
        refresh_btn.click(refresh_files, inputs=dir_tb, outputs=file_dd)
        dir_tb.change(refresh_files, inputs=dir_tb, outputs=file_dd)
        load_btn.click(load_selected, inputs=[dir_tb, file_dd], outputs=load_outputs)
        upload.upload(load_uploaded, inputs=upload, outputs=load_outputs)
        preview_btn.click(preview_config, inputs=edit_inputs, outputs=[preview, status])
        save_btn.click(save_config, inputs=[dir_tb, save_name] + edit_inputs,
                       outputs=[status, file_dd])

    return demo


if __name__ == "__main__":
    build_ui(default_dir=".").launch()
