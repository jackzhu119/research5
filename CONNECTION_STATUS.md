# Colab connection status

Verified 2026-11-17, Asia/Shanghai. No OAuth link, token, or callback from a previous cloud environment was copied. Credentials are excluded from this repository.

## Local official browser bridge: execution verified

- The Codex desktop app and command executor were observed running on this Windows 11 computer. uvx was already installed (WinGet); installing uv was unnecessary.
- Backed up the existing local MCP configuration before adding an independent `colab-browser` STDIO entry: command `uvx`, args `["git+https://github.com/googlecolab/colab-mcp"]`. Other server entries were retained. The backup remains private on the local computer and is not committed.
- Official bridge source actually resolved to commit `b9ab3899e0f1fa493390b1fd6d54aa2e464ecdf1`. It launched as a local uvx/Colab-MCP/Python process and listened on loopback.
- The initial advertised tool was `open_colab_browser_connection` with an empty input schema. Called this actual tool with `{}`; it opened the local browser. The first attempt timed out while login took longer than the official 60-second wait. After restarting the bridge and the user's login/connection, it returned `true`.
- At 2026-11-17 16:37:58 China time, the local diagnostic client received **`notifications/tools/list_changed`**. Subsequent tools/list returned add_code_cell, add_text_cell, get_cells, run_code_cell, update_cell, move_cell, delete_cell and the connection tool.
- Added a Python cell containing `print("COLAB_LOCAL_OK")`, ran it via run_code_cell, then independently read it with get_cells(includeOutputs=true). Both returned stdout **COLAB_LOCAL_OK** with execution_count=1. The local logs preserve actual tool specifications, execution and readback.

Client compatibility limit: registration did not hot-load a new native tool namespace into this already-running desktop chat. The end-to-end test therefore used the official FastMCP Python SDK as a local STDIO diagnostic client controlled from this Codex session. That client demonstrably received dynamic tool changes. This proves the local bridge and notebook edit/execute/read path; it does **not** prove native ChatGPT Work hot reload or native notification handling. No claim of guaranteed Work compatibility is made. Browser UI automation also stopped because its policy check could not determine the current URL; subsequent notebook operations used MCP, not UI automation.

## Official remote MCP: separate session verified

- Endpoint: https://colaboratory.googleapis.com/mcp, via the already-configured local connector. Authentication was valid in this session; there was no need to copy or replay a cloud authentication flow.
- get_subscription returned SUBSCRIPTION_TIER_FREE; list_runtime_specs returned standard CPU and standard T4 eligible; list_runtimes initially returned no runtime.
- T4 creation returned service unavailable. Rechecking listed no runtime. Standard CPU creation succeeded without buying a plan or enabling additional billing.
- Runtime `runtimes/research5-local-pilot-cpu`; independent session `research5-first-offline-validation`. execute_code returned COLAB_LOCAL_OK, executionCount=1.
- Initial environment: Python 3.13.18, PyTorch 2.14.1+cpu, no CUDA/GPU, about 12.7 GiB total RAM. Full environment and actual dependency versions are in results/stage-a-v2/environment.json and pip-freeze.txt.
- Scripts, sources and manifests were transferred using the documented authenticated runtime HTTP file interface. Data/model files and completed results were downloaded back to the local research5 directory.

The remote session is separate from the browser notebook. Runtime/session execution does not establish attachment to any browser notebook. The browser notebook was independently verified above.

## Durable background execution

The local bridge disconnects when the computer shuts down. Colab runtimes can expire and are not a durable unattended job service. In response to the user's explicit request, an independent GitHub Actions workflow is being deployed to the public research5 repository using standard hosted CPU runners. It needs no Google OAuth credentials and writes artifacts even on failure. Actual deployment/run status and links must be read from PILOT_REPORT.md and the repository Actions page; configuration alone is not proof of a running job.

Official setup sources: [browser bridge README](https://github.com/googlecolab/colab-mcp), [remote MCP reference](https://developers.google.com/colab/api/reference/mcp).
