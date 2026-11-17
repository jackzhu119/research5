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

## Google Drive persistence and the final execution destination

The user's final choice is **Colab compute plus Google Drive storage**, with GitHub retaining code. The earlier Actions job actually started but failed dependency resolution (YASA requires pandas>=2.3); failure artifacts were saved. Actions is now a manual-only backup and is not the selected compute path.

Google Drive folder: [research5](https://drive.google.com/drive/folders/14k_7KUZu535k0oKM7SDWisLK7HWvifBO). Stage-A raw data/model/result archive (48,946,810 bytes) and Stage-B paired/oracle archive (11,407,347 bytes) were uploaded with connector metadata readback. The browser Colab runtime mounted Drive through the user's normal authorization. An executed cell wrote and read `COLAB_DRIVE_OK.txt`; the independent Drive connector found that file inside the same folder, confirming that the mounted account and connected destination match.

`pilot/colab_drive_run.py` uses an isolated scientific environment (pandas3.1.1 rather than changing the notebook kernel), checks dependency compatibility, keeps data/models/results/logs on Drive, records progress and checksums, resumes verified completed phases, and preserves partial failures. Compatible single-record A/B gates precede the frozen limited development queue. The local computer is not required after the code cell starts in Colab. This is not a guarantee of perpetual execution: normal Colab runtime reclamation or quotas can interrupt it, and a reclaimed runtime cannot automatically reauthorize Drive. State/checkpoints survive on Drive; a later authorized runtime can resume.

Actual start was verified: `colab-drive-20261117-v2`, 2026-11-17 17:20:40 Asia/Shanghai. Drive independently returned RUNNING/stage-a and the compatible isolated environment record. The pipeline's own source ID is `1b7b69e9a79ca03e1be5`; source bundle SHA256 is `62c7991a2a6c600e162656af591126264b057cbbedcb84b1a0a537da20538c07`. See [live status folder](https://drive.google.com/drive/folders/1i-q-8btlTDvBx9b7M4SqL5BU4Q-B6nvs). This verifies an actual server-side start, not completion of all queued work. The earlier v1 ensurepip bootstrap failure is retained separately.

At 17:23:44 Asia/Shanghai, independent Drive readback advanced to RUNNING/stage-b, verifying automatic transition after the compatible stage-A gate. No local next-step command was needed. The queue remains pending its upstream gates; normal runtime limits apply.

Official setup sources: [browser bridge README](https://github.com/googlecolab/colab-mcp), [remote MCP reference](https://developers.google.com/colab/api/reference/mcp).
