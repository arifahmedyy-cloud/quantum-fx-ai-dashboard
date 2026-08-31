"""Regression test for a confirmed audit bug: Streamlit's `st.session_state`
is bound to the ScriptRunContext of whichever thread is executing the
script. `threading.Thread(target=trading_loop, ...).start()` spawns a plain
thread with NO context attached, so every `st.session_state.xxx` access
inside that thread raised `AttributeError: st.session_state has no
attribute "..."` — this was reproduced with the exact error message from a
real user's terminal (screenshot during the audit) and confirmed here with
the REAL `streamlit` package (not a mock), via Streamlit's own `AppTest`
runtime so a genuine ScriptRunContext exists.

Fixed by adding `app._start_background_thread()`, which uses
`streamlit.runtime.scriptrunner.add_script_run_ctx` to copy the calling
thread's context onto the new thread before starting it.

This runs in an isolated subprocess (not in-process) because the rest of
the test suite's tests/conftest.py installs a lightweight FAKE `streamlit`
module into sys.modules for every other test — this test needs the real
package, so it can't share the same Python process/sys.modules.
"""
import os
import subprocess
import sys
import textwrap

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PROBE_SCRIPT = textwrap.dedent(f"""
    import sys, os
    sys.path.insert(0, {REPO_ROOT!r})
    import streamlit as st
    import app as qfx_app

    if "broker" not in st.session_state:
        st.session_state.broker = "REAL_BROKER_OBJECT"
    if "outcome" not in st.session_state:
        st.session_state.outcome = None

    def worker():
        # Exactly what trading_loop's first line does: read a value out of
        # st.session_state from inside a background thread.
        st.session_state.outcome = st.session_state.broker

    t = qfx_app._start_background_thread(worker)
    t.join(timeout=5)

    st.write("outcome:" + str(st.session_state.outcome))
""")


def test_background_thread_can_access_session_state(tmp_path):
    """Runs the probe script through Streamlit's real AppTest runtime in a
    fresh subprocess (clean sys.modules, real `streamlit` package) and
    asserts the background thread successfully read st.session_state.
    """
    script_path = tmp_path / "probe.py"
    script_path.write_text(PROBE_SCRIPT)

    runner = textwrap.dedent(f"""
        import sys
        sys.path.insert(0, {REPO_ROOT!r})
        from streamlit.testing.v1 import AppTest
        at = AppTest.from_file({str(script_path)!r})
        at.run(timeout=15)
        if at.exception:
            print("EXCEPTION:" + str(at.exception))
            sys.exit(1)
        texts = [el.value for el in at.markdown]
        print("MARKDOWN:" + "|".join(texts))
    """)
    runner_path = tmp_path / "runner.py"
    runner_path.write_text(runner)

    result = subprocess.run(
        [sys.executable, str(runner_path)],
        capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, (
        f"Background thread could not access st.session_state — the "
        f"add_script_run_ctx fix has regressed.\nstdout: {result.stdout}\n"
        f"stderr: {result.stderr}"
    )
    assert "outcome:REAL_BROKER_OBJECT" in result.stdout, result.stdout
