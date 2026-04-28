const { spawn } = require("child_process");
const fs = require("fs");
const path = require("path");

const cwd = __dirname;
const pythonExe =
  "C:\\Users\\Deepak\\.cache\\codex-runtimes\\codex-primary-runtime\\dependencies\\python\\python.exe";

const out = fs.openSync(path.join(cwd, "streamlit-node.out.log"), "a");
const err = fs.openSync(path.join(cwd, "streamlit-node.err.log"), "a");

const child = spawn(
  pythonExe,
  ["-m", "streamlit", "run", "app.py", "--server.port", "8501", "--server.headless", "true"],
  {
    cwd,
    detached: true,
    stdio: ["ignore", out, err],
    windowsHide: true,
  }
);

child.unref();
console.log(child.pid);
